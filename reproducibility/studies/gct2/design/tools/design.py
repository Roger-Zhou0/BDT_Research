"""Design manifest: CPU-only, no training, evaluation, or account mutation."""
from __future__ import annotations
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import hashlib, json

ARMS = ('adamw', 'bdt', 'scrambled_matched')
SEEDS = tuple(range(161, 167))  # Fixed candidates; a complete live inventory gate is mandatory.
ORDER = {
    161: ('adamw', 'bdt', 'scrambled_matched'),
    162: ('adamw', 'scrambled_matched', 'bdt'),
    163: ('bdt', 'scrambled_matched', 'adamw'),
    164: ('bdt', 'adamw', 'scrambled_matched'),
    165: ('scrambled_matched', 'adamw', 'bdt'),
    166: ('scrambled_matched', 'bdt', 'adamw'),
}
CACHE = {
    'metadata.json': '64d9e563d8b12517cfe123560d39ab4f579ef9c4ec5482e215c20cb4f4108feb',
    'train.bin': '760fe8d0a55959dc322643fa4ac856e542e11bd59ac84c2f48f6e00a3d587ce6',
    'val.bin': '5dd2caeb46e3c237e6c9291d5964c548ae004cd87b3c29f01188b444af2aff0f',
}
GCT1_SOURCE = 'cb703e1fb6f619ae7044cb47807483be375f47b4dbeb7c7a75af09b7e6ad5048'
GCT1_PROTOCOL = 'f3cd9097f66360d4e71a5932c780d3e216520f7a29222f6fe4a82bb44a975298'
REFERENCE_TEMPER_SHA = 'd1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1'
OCB1_SOURCE = 'e3a4d782bb662ce6eec7674aca00ce16304aa29d87673f101dbbfe4392b80d99'
CMS1_PROTOCOL_FILE_SHA = 'e941b4274b27b1642e9467fdafe144fa79c378c199d2b5d35358fb7428261b7c'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def budget():
    parent = 7630
    post = 3000
    prefix = 200
    per_seed = parent + prefix + len(ARMS) * (post - prefix)
    # FP32 model + two FP32 moments; conservative payload estimate, not measured usage.
    retained_bytes = 6 * 12 * (81074688 + 11 * 123551232)
    return dict(parent_updates=6*parent, shared_prefix_updates=6*prefix,
                branch_updates=18*(post-prefix), total_scientific_updates=6*per_seed,
                total_campaign_token_exposures=6*per_seed*32768,
                recommended_path_token_exposures=(parent+post)*32768,
                parent_cost_per_path_updates=parent, post_cost_per_path_updates=post,
                parent_parameters=81074688, target_parameters=123551232,
                approximate_retained_payload_GiB=retained_bytes/2**30,
                initial_cohort_free_disk_GiB=128, per_GPU_free_memory_GiB=32,
                budget_scope='Mixed small/large optimizer updates, NOT homogeneous FLOPs or wall-clock.')

GCT2 = dict(
    name='GCT2', version='gct2_design_v1', status='DESIGN_COMPLETE_NOT_REGISTERED_NOT_LAUNCHED',
    purpose='Confirm fixed-recipe pretrained-growth utility and own-state direction/strength separation.',
    proposed_root='runs/growth_correspondence_confirmation_v1',
    candidate_seeds=list(SEEDS), seed_policy='All six must pass a live complete inventory check; collision or incomplete audit stops. No silent replacement.',
    pilot_seeds_excluded=[159, 160], statistical_unit='Independent small-parent training seed',
    lanes={'B-good': {'account':'account_b','session':'good','seeds':[161,163,165]},
           'B-default': {'account':'account_b','session':'default','seeds':[162,164,166]}},
    cache_label='S1-B', cache_sha256=CACHE, data_directory='data_s1/fineweb_edu_3b',
    parent=dict(layers=6, parameters=81074688, steps=7630, peak_lr=.0008),
    growth=dict(mapping=list(range(6))*2, copy='all block weights, embeddings, final norm, both named Adam moments, optimizer age',
                independent_storage=True, function_preserving=False,
                preserved_global_age=7630, next_training_data_step=7631,
                reset_only=['post-growth LR phase','both whole-depth covariance histories']),
    post=dict(layers=12,parameters=123551232,total_phase_updates=3000,peak_lr=.0006,shared_prefix_updates=200),
    shape=dict(d_model=768,heads=12,ffn_hidden=2048,vocab_size=50257,context=512,microbatch=16,accumulation=4),
    training=dict(precision='bf16',tokens_per_update=32768,warmup=200,min_lr_ratio=.1,
                  beta1=.9,beta2=.95,eps=1e-8,decay=.1,gradient_clip=1.),
    bdt=dict(cap=3.,observe_every=10,covariance_beta=.9,warmup=200,ramp=200,scope='whole twelve-depth FFN gate/up/down'),
    matching=dict(reference='CMS1 own-state matching, unchanged in GCT1',compute='FP64 geometry, FP32 output cast',
        zero_angle_sine=1e-7,minimum_candidate_sine=1e-7,norm_relative_tolerance=1e-5,
        common_relative_tolerance=1e-5,angle_absolute_tolerance=5e-6,
        displacement_absolute_tolerance=2e-6,displacement_relative_tolerance=.0002,
        random_fallback=False,log_all_degeneracies=True),
    arms=list(ARMS), arm_order={str(k):list(v) for k,v in ORDER.items()},
    shadows='All three scientific arms retain both hypothetical histories and matching construction; only applied FFN proposals differ.',
    strict=dict(deterministic=True,warn_only=False,cudnn_deterministic=True,cudnn_benchmark=False,
                matmul_tf32=False,workspace=':4096:8',attention='automatic, record actual dispatch'),
    dev=dict(offset=0,sequences=2048,context=512,scored_tokens=1048576,stop=1050624),
    report=dict(offset=2500000,sequences=2048,context=512,scored_tokens=1048576,stop=3550624,
                when='after all 18 final training records and checkpoints are verified and the cohort lock is committed',
                novelty='Established S1-B REPORT text, NOT newly unseen data'),
    analysis=dict(contrasts=[['bdt','scrambled_matched'],['bdt','adamw']],n=6,ordinary_confidence=.95,
                  adjusted_confidence=.975,method='Paired Student-t, df=5; Bonferroni family of two contrasts',
                  t975_df5=2.5705818356363146,t9875_df5=3.163381449748609,
                  normality_scope='Nominal small-sample t intervals assume approximately normal independent seed differences.',
                  joint_success='Both adjusted upper bounds < 0, after all validity checks; report both estimates regardless.',
                  optional_stopping=False,adapt_n=False,zero_variance='Flag for review; do not auto-declare confirmation'),
    checkpoint=dict(every=250,keep_post=[0,200,1000,2000,3000],atomic=True),
    retries=dict(max_total_same_recipe_OOM_attempts_per_job=3,nonfinite='stop',invariant='stop',
                 code_or_data_change='stop',other_interruption='explicit resume, no recipe/seed replacement'),
    release=dict(parent_reference_source_digest=GCT1_SOURCE,parent_reference_protocol_digest=GCT1_PROTOCOL,
                 train_loop='unchanged arithmetic from GCT1; new scheduler and REPORT evaluator require independent tests',
                 allow_BCO1_candidate_in_confirmation=False),
    no_auto_followups=True, budget=budget(),
)

CONDITIONS=('split_adam','reference_bdt','candidate_bdt')
BCO1=dict(
    name='BCO1',version='bco1_design_v1',status='DESIGN_COMPLETE_CPU_TRANSFORM_PROTOTYPE_ONLY',
    target=dict(account='account_a',session='good'),engineering_seed=127,
    question='Can avoidable implementation work be removed without changing the measured treatment?',
    frozen_reference=dict(temper_sha256=REFERENCE_TEMPER_SHA,ocb1_source_digest=OCB1_SOURCE),
    unchanged=['cap','observation cadence','covariance EMA','scope','onset','ramp','FP32/FP64 arithmetic','normalization','learning-rate schedule','optimizer moments'],
    candidate='diagnostics_lite_v1: omit raw-update cosine and common-error reporting only when diagnostics=False; preserve norm/condition safety checks and update arithmetic',
    candidate_policy='At most one versioned implementation candidate. No quality tuning. Any substantive second implementation requires a new scope decision.',
    profile=dict(update_range=[501,520],refresh_steps=[510,520],ordinary_steps=18,
        include=['proposal formation','centering/projection','covariance formation','eigendecomposition','operator application','norm restoration','safety checks','optional reporting','apply/decay','host synchronization','allocations'],
        runtime_events='Use disjoint record_function regions in a separate instrumentation copy; retain CPU/CUDA event keys and scalar-sync events.',
        report='Refresh vs ordinary separately; exclusive time where available. Nested or overlapping durations are not additive.',
        primary_timers_profiled=False),
    validation=dict(cpu_transform_steps=1000,full_size_common_prefix_end=200,active_end=1000,
        paths=['reference','candidate_uninterrupted','candidate_interrupt310_resume'],max_model_updates=2600,
        required_exact=['raw proposals','applied proposals','moments and counters','model tensors','covariance/eigen state','RNG','training batches'],
        report_fields_allowed_to_differ=['raw_update_cosine','common_max_abs_error','timings'],
        multiple_seeds='engineering seed127 only; no scientific seed consumption'),
    timing=dict(conditions=list(CONDITIONS),main_blocks=12,orders=[list(x) for x in permutations(CONDITIONS)]*2,
        families=['full_step','optimizer_only'],full_step_updates=20,optimizer_only_updates=100,
        warmup_updates=20,sentinels=['split_adam_before','split_adam_after'],sentinel_drift_limit=.1,
        minimum_stable_blocks=8,restore_after_warmup=True,one_gpu_uuid=True,
        source_state='Hash-verified OCB1 Dense12 engineering update500 snapshot; rebuild only if recorded recipe/state identity matches.',
        full_step_input='Same fixed CPU batch ring; H2D, forward/backward, clipping and optimizer included.',
        optimizer_input='Same fixed clipped gradients. Include two or ten covariance refreshes in 20/100 updates respectively.',
        summary='Median paired block ratios + IQR; all completed blocks primary, stable subset secondary',
        comparison=['candidate/reference','reference/split','candidate/split'],
        exclude=['profiling','validation','checkpoint I/O','audit hashes','one-off state restore'],
        exact_final_states_must_agree=True),
    engineering_success=dict(minimum_median_full_step_reduction=.02,minimum_faster_blocks=10,
        required_quality_checks='all exact-equivalence checks plus >=8 stable full-step blocks',
        overhead_goal='Halve residual overhead versus split Adam in the contemporaneous reference/candidate measurement; a goal, not a publication gate.'),
    restrictions='Do not change GCT2, overwrite OCB1, aggregate across devices, substitute profile duration for timing, or claim time-to-quality.',
    stages=['profile','one_candidate','exact_equivalence_gate','balanced_before_after_measurement','stop_and_report'],
)

def jobs():
    result=[]
    for seed in SEEDS:
        lane='B-good' if seed%2 else 'B-default'
        stem=f'seed_{seed}'
        result.extend([
            dict(id=stem+'/parent',seed=seed,lane=lane,stage='parent',arm='adamw',start=0,end=7630,updates=7630,depends_on=[]),
            dict(id=stem+'/grown',seed=seed,lane=lane,stage='grow',arm='adamw',start=0,end=0,updates=0,depends_on=[stem+'/parent']),
            dict(id=stem+'/prefix',seed=seed,lane=lane,stage='prefix',arm='adamw',start=0,end=200,updates=200,depends_on=[stem+'/grown']),
        ])
        for arm in ORDER[seed]:
            result.append(dict(id=stem+'/arms/'+arm,seed=seed,lane=lane,stage='arm',arm=arm,start=200,end=3000,updates=2800,depends_on=[stem+'/prefix']))
    return result

def engineering_key(lane,gpu_uuid,runtime_digest,source_digest):
    for item in (lane,gpu_uuid,runtime_digest,source_digest):
        if not item or '/' in item or '..' in item:raise ValueError('Invalid engineering key component')
    return f'engineering/{lane}/{gpu_uuid}/{runtime_digest}/{source_digest}'

def write_manifests(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    values={'GCT2.json':GCT2,'GCT2_JOBS.json':jobs(),'BCO1.json':BCO1,'BUDGET.json':budget()}
    for name,obj in values.items():
        (directory/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return {name:digest(obj) for name,obj in values.items()}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'protocols')
    args=ap.parse_args();print(json.dumps(write_manifests(args.output),indent=2))
