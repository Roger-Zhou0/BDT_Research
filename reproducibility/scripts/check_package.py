"""Verify shipped bytes, native kernel provenance, syntax, and basic anonymity hygiene.

This scanner is a release check, not a guarantee of anonymity. Author-side
manual review and data/rights verification remain separate release tasks.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,re,sys,io,tokenize
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def privacy_patterns():
    # Patterns are written compositionally to avoid matching the scanner's own source.
    patterns={
      'absolute_home':re.compile(r'/(?:Us'+'ers|ho'+r'me)/[^\s/]+/'),
      'email':re.compile(r'[A-Za-z0-9._%+-]+@' + r'[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
      'credential_marker':re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE' + r' KEY-----'),
      'credential_assignment':re.compile(r'(?i)(?:api_key|access_token|password)\s*[:=]\s*[\"\x27][^\"\x27\s]{8,}'),
      'private_notebook_url':re.compile(r'https?://[^\s]+/' + r'user/[^\s]+'),
    }
    return patterns


def check():
    path=ROOT/'MANIFEST_SHA256.json'
    manifest=json.loads(path.read_text());errors=[]
    allowed_suffixes={'.py','.json','.md','.txt','.csv','.toml','.tex','.yaml','.sh','.jsonl'}
    forbidden_suffixes={'.pt','.pth','.bin','.pem','.key','.env','.log','.sqlite','.db','.pyc','.zip'}
    patterns=privacy_patterns()
    scanned=0
    for rel,h in manifest.items():
        p=ROOT/rel
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):errors.append('unsafe path: '+rel);continue
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:errors.append('hash mismatch: '+rel);continue
        # One small original design-test transcript is an explicitly retained source-manifest member.
        approved_record = rel == 'studies/gct2/design/evidence/DESIGN_TESTS.log'
        if (p.suffix in forbidden_suffixes or p.suffix not in allowed_suffixes) and not approved_record:errors.append('unapproved file type: '+rel)
        if p.suffix=='.py':
            try:ast.parse(p.read_text())
            except SyntaxError as e:errors.append('syntax: '+rel+':'+str(e))
        text=p.read_text()
        email_text=text
        if p.suffix=='.py':
            # Tensor expressions such as left.T @ right.basis are not email addresses.
            email_text='\n'.join(token.string for token in tokenize.generate_tokens(io.StringIO(text).readline)
                                  if token.type in (tokenize.STRING,tokenize.COMMENT))
        for label,pattern in patterns.items():
            if pattern.search(email_text if label=='email' else text):errors.append(label+': '+rel)
        scanned+=1
    tracked=set(manifest)|{'MANIFEST_SHA256.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.relative_to(ROOT).parts and '.venv' not in p.relative_to(ROOT).parts}
    for rel in sorted(actual-tracked):errors.append('unmanifested: '+rel)
    for p in ROOT.rglob('*'):
        if any(x in p.parts for x in ('.git','.env','.DS_Store')):errors.append('forbidden metadata: '+str(p.relative_to(ROOT)))
    native=json.loads((ROOT/'native/NATIVE_KERNEL_MANIFEST.json').read_text())
    for rel,h in native.items():
        if hashlib.sha256((ROOT/'native'/rel).read_bytes()).hexdigest()!=h:errors.append('native kernel changed: '+rel)
    origin=json.loads((ROOT/'native/ORIGIN.json').read_text())
    helper=(ROOT/'repro/native_training_helpers.py').read_text(); lines=helper.splitlines(); checked_excerpts=0
    for node in ast.parse(helper).body:
        if isinstance(node,ast.FunctionDef) and node.name in origin['function_excerpts']:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            excerpt='\n'.join(lines[start-1:node.end_lineno])+'\n'
            if hashlib.sha256(excerpt.encode()).hexdigest()!=origin['function_excerpts'][node.name]:errors.append('native function excerpt changed: '+node.name)
            checked_excerpts+=1
    if checked_excerpts!=len(origin['function_excerpts']):errors.append('missing native function excerpt')
    archived=json.loads((ROOT/'reference/ARCHIVED_DATA_AND_RUNTIME.json').read_text())
    preparation=ROOT/'reference/data/prepare_s1_data.py.txt'
    if hashlib.sha256(preparation.read_bytes()).hexdigest()!=archived['prepare_source_sha256']:
        errors.append('original preparation source changed')
    try:ast.parse(preparation.read_text())
    except SyntaxError:errors.append('original preparation source syntax')
    report_origin=json.loads((ROOT/'reference/cms1/ORIGIN.json').read_text())
    expected_reports={f'seed_{seed}/{arm}_report.json' for seed in range(147,153)
                      for arm in ('adamw','bdt','scrambled_matched')}
    bindings=report_origin['bindings']; archived_reports_checked=0
    if len(bindings)!=18 or {row['path'] for row in bindings}!=expected_reports:
        errors.append('incomplete archived CMS1 REPORT cohort')
    for row in bindings:
        if row['path'] not in expected_reports:continue
        raw=(ROOT/'reference/cms1'/row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            errors.append('archived CMS1 report changed: '+row['path'])
        else:archived_reports_checked+=1
    return {'passed':not errors,'manifest_files':len(manifest),'scanned_text_files':scanned,'native_function_excerpts_checked':checked_excerpts,
            'native_byte_identical_files':len(native),'archived_cms1_reports_checked':archived_reports_checked,
            'original_preparation_source_checked':True,'errors':errors,
            'boundary':'Text/path checks only; not a claim of guaranteed anonymization, source licensing, GPU replication, or byte-level data availability.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args();r=check();print(json.dumps(r,indent=2))
    if not r['passed']:sys.exit(1)

if __name__=='__main__':main()
