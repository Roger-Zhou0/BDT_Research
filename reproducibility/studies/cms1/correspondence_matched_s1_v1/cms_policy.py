"""Same frozen split AdamW in every arm; only the private-FFN proposal differs."""
from __future__ import annotations
import copy
import torch
import cms_protocol as p
from cms_geometry import make_permutations, validate_permutations, transport, match_direction
from cms_runtime import cpu_tree
PROJECTIONS = ('gate', 'up', 'down')

class Policy:

    def __init__(self, model, cfg, mode, device, backend, seed):
        if mode not in (*p.ARMS, 'reference_adamw', 'reference_bdt'):
            raise ValueError('Unknown policy ' + mode)
        self.mode = mode
        self.backend = backend
        self.device = device
        fm = 'temper_cap3' if mode == 'reference_bdt' else 'adamw'
        self.frozen = backend['optimizers'].PolicyOptimizer(model, cfg, fm, device)
        self.base, self.rest, self.groups = (self.frozen.base, self.frozen.rest, self.frozen.groups)
        self.by_name = dict(model.named_parameters())
        self.names = {id(t): n for n, t in self.by_name.items()}
        self.group_names = {q: [self.names[id(t)] for t in ps] for q, ps in self.groups.items()}
        grouped = {id(t) for ps in self.groups.values() for t in ps}
        self.rest_names = [n for n, t in self.by_name.items() if id(t) not in grouped]
        roles, hidden = (len(self.groups['gate']), self.groups['gate'][0].shape[0])
        self.permutations = make_permutations(roles, hidden, seed).to(device)
        validate_permutations(self.permutations)
        ref = backend['tempering_reference']
        tc = ref.TemperConfig(200, 200, 10, 0.9, 3.0)
        self.canonical = self.frozen.tempers if mode == 'reference_bdt' else {q: ref.OnlineDepthTemper(roles, device, torch.float32, tc) for q in PROJECTIONS}
        self.scrambled = {q: ref.OnlineDepthTemper(roles, device, torch.float32, tc) for q in PROJECTIONS}
        self.last_reports = {}
        self.summary = dict(active_groups=0, matched_groups=0, numerical_zero_groups=0, exact_identity_groups=0, max_angle_error=0.0, max_norm_error=0.0, max_common_error=0.0, max_displacement_error=0.0, max_amplification=0.0)
        self._frozen_parameter_names_for_restore = frozenset((n for n, parameter in model.named_parameters() if not parameter.requires_grad))

    @torch.no_grad()
    def step(self, lr, step, diagnostic=False):
        if self.mode.startswith('reference_'):
            return self.frozen.step(lr, step, diagnostic=diagnostic)
        if self.base.step_num != step - 1:
            raise RuntimeError('Adam age/update mismatch')
        raw = self.base.propose(lr)
        applied = {}
        reports = {}
        for q, u in raw.items():
            if q not in PROJECTIONS:
                applied[q] = u
                continue
            canonical, c_rep = self.canonical[q].transform(u.float(), step, active=True)
            vperm, s_rep = self.scrambled[q].transform(transport(u, self.permutations, q), step, active=True)
            mapped = transport(vperm, self.permutations, q, inverse=True)
            matched, metrics = match_direction(u, canonical, mapped)
            if abs(c_rep.get('centered_norm_ratio', 1.0) - 1.0) > 5e-05 or c_rep['operator_condition'] > 3.00005:
                raise RuntimeError('Canonical frozen BDT invariant failed')
            if self.mode == 'adamw':
                out = u
            elif self.mode == 'bdt':
                out = canonical
            else:
                out = matched
            if step <= p.PREFIX and (not torch.equal(out, u)):
                raise RuntimeError('Warmup unexpectedly changed an applied proposal')
            applied[q] = out
            reports[q] = dict(canonical=c_rep, scrambled=s_rep, matching=metrics)
            if step > p.PREFIX:
                self.summary['active_groups'] += 1
                kind = metrics['kind']
                if kind == 'matched':
                    self.summary['matched_groups'] += 1
                elif kind == 'numerical_zero':
                    self.summary['numerical_zero_groups'] += 1
                else:
                    self.summary['exact_identity_groups'] += 1
                for name in ('angle_error', 'norm_error', 'common_error', 'displacement_error', 'amplification'):
                    value = metrics.get(name)
                    if value is not None:
                        key = 'max_' + name
                        self.summary[key] = max(self.summary[key], float(value))
        self.base.apply(lr, applied)
        for group in self.rest.param_groups:
            group['lr'] = lr
        self.rest.step()
        self.last_reports = reports
        return reports if diagnostic else {}

    def adam_state(self):
        states = {}
        for q, params in self.groups.items():
            for t in params:
                st = self.base.state[id(t)]
                states[self.names[id(t)]] = dict(kind='projection', m=st['m'], v=st['v'])
        for name in self.rest_names:
            states[name] = dict(kind='rest', state=self.rest.state.get(self.by_name[name], {}))
        groups = []
        for g in self.rest.param_groups:
            groups.append({k: copy.deepcopy(v) for k, v in g.items() if k != 'params'})
        return dict(age=self.base.step_num, group_names=self.group_names, rest_names=self.rest_names, states=states, rest_groups=groups, hyper={k: getattr(self.base, k) for k in ('beta1', 'beta2', 'eps', 'weight_decay')})

    @torch.no_grad()
    def load_adam(self, state):
        if state['group_names'] != self.group_names or state['rest_names'] != self.rest_names:
            raise RuntimeError('Named optimizer partition changed')
        if state['hyper'] != {k: getattr(self.base, k) for k in state['hyper']}:
            raise RuntimeError('Optimizer hyperparameters changed')
        if set(state['states']) != set(self.by_name):
            raise RuntimeError('Optimizer coverage changed')
        age = int(state['age'])
        self.base.step_num = age
        for params in self.groups.values():
            for t in params:
                row = state['states'][self.names[id(t)]]
                if row['kind'] != 'projection':
                    raise RuntimeError('Projection state mislabeled')
                for k in ('m', 'v'):
                    value = row[k]
                    if value.shape != t.shape or value.dtype != t.dtype or (not bool(torch.isfinite(value).all())):
                        raise RuntimeError('Invalid Adam moment ' + self.names[id(t)] + '/' + k)
                    self.base.state[id(t)][k].copy_(value.to(t.device))
        for name in self.rest_names:
            t = self.by_name[name]
            row = state['states'][name]
            if row['kind'] != 'rest':
                raise RuntimeError('Rest state mislabeled')
            self.rest.state[t] = {}
            for k, value in row['state'].items():
                if torch.is_tensor(value):
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite rest state')
                    if k in ('exp_avg', 'exp_avg_sq') and value.shape != t.shape:
                        raise RuntimeError('Invalid rest moment shape')
                    dev = t.device if k != 'step' or self.rest.defaults.get('fused', False) else 'cpu'
                    value = value.to(dev).clone()
                self.rest.state[t][k] = value
            if name not in self._frozen_parameter_names_for_restore and (age and (not self.rest.state[t] or int(self.rest.state[t]['step']) != age)):
                raise RuntimeError('Rest Adam step mismatch for ' + name)
        if len(state['rest_groups']) != len(self.rest.param_groups):
            raise RuntimeError('Rest groups changed')
        for old, new in zip(state['rest_groups'], self.rest.param_groups):
            for k, v in old.items():
                if k != 'lr' and new.get(k) != v:
                    raise RuntimeError('Rest optimizer setting changed: ' + k)
            new['lr'] = old['lr']

    def state_dict(self):
        return cpu_tree(dict(adam=self.adam_state(), canonical={q: x.state_dict() for q, x in self.canonical.items()}, scrambled={q: x.state_dict() for q, x in self.scrambled.items()}, permutations=self.permutations, summary=self.summary), torch)

    @torch.no_grad()
    def load_state_dict(self, state):
        self.load_adam(state['adam'])
        if not torch.equal(self.permutations.cpu(), state['permutations'].cpu()):
            raise RuntimeError('Permutation seed/state mismatch')
        for field in ('canonical', 'scrambled'):
            objects = getattr(self, field)
            if set(objects) != set(state[field]):
                raise RuntimeError('Covariance groups changed')
            for q, t in objects.items():
                t.load_state_dict(state[field][q])
                if not self.mode.startswith('reference_') and t.last_step != self.base.step_num:
                    raise RuntimeError('Covariance/Adam age mismatch')
        self.summary = copy.deepcopy(state['summary'])

    def identity(self, ms):
        return dict(adam=ms.state_tree_hash(self.adam_state()), canonical=ms.state_tree_hash({q: t.state_dict() for q, t in self.canonical.items()}), scrambled=ms.state_tree_hash({q: t.state_dict() for q, t in self.scrambled.items()}), permutations=ms.state_tree_hash(self.permutations))
