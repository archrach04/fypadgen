"""Quick test of the Two-Stage Ad Layout System."""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

# Extract and exec just the class code from app.py
exec_globals = {'__builtins__': __builtins__}
exec('import numpy as np', exec_globals)
exec('BANNER_DEBUG = True', exec_globals)
exec('ELEMENT_TYPES = ["logo 0", "text 1", "text 2", "underlay 3"]', exec_globals)

with open(os.path.join(os.path.dirname(__file__), 'app.py'), 'r', encoding='utf-8') as f:
    src = f.read()

start = src.find('class PlacementPlanner:')
end = src.find('def _vlm_analyze_image_for_banner')
class_code = src[start:end]
exec(class_code, exec_globals)

PlacementPlanner = exec_globals['PlacementPlanner']
LayoutRenderer = exec_globals['LayoutRenderer']
AdLayoutOrchestrator = exec_globals['AdLayoutOrchestrator']
run_two_stage = exec_globals['run_two_stage_ad_layout']

# Test data
detected = [
    {'label': 'product', 'bbox': [100, 80, 280, 400], 'importance': 'high'},
    {'label': 'face',    'bbox': [180, 90,  60,  70], 'importance': 'high'},
    {'label': 'bg_obj',  'bbox': [10, 500, 100,  80], 'importance': 'low'},
]

print('='*60)
print('  TWO-STAGE AD LAYOUT SYSTEM TEST')
print('='*60)
result = run_two_stage(480, 640, detected)

print('\n=== STAGE 1: PLACEMENT PLAN (text only, no coords) ===')
print(result['placement_plan'])

print('\n=== STAGE 2: RESOLVED ELEMENTS (pixel coords) ===')
for el in result['elements']:
    print(f"  {el['element']:12s}  role={el['role']:8s}  "
          f"left={el['left']:4d}  top={el['top']:4d}  "
          f"w={el['width']:4d}  h={el['height']:4d}  z={el['z_index']}")

print('\n=== STAGE 2: HTML OUTPUT ===')
print(result['html'])

# Validation checks
plan = result['placement_plan']
assert 'Placement Plan:' in plan, 'Plan header missing'
assert 'logo 0' in plan.lower(), 'Logo missing from plan'
assert 'text 1' in plan.lower(), 'Text 1 missing from plan'
assert 'text 2' in plan.lower(), 'Text 2 missing from plan'
assert 'underlay 3' in plan.lower(), 'Underlay missing from plan'

html = result['html']
assert '<html>' in html, 'HTML tag missing'
assert 'class="canvas"' in html, 'Canvas div missing'
assert 'class="logo"' in html, 'Logo div missing'
assert 'class="text"' in html, 'Text div missing'
assert 'class="underlay"' in html, 'Underlay div missing'

for el in result['elements']:
    assert el['left'] >= 0, f"{el['element']} left < 0"
    assert el['top'] >= 0, f"{el['element']} top < 0"
    assert el['left'] + el['width'] <= 480, f"{el['element']} exceeds right edge"
    assert el['top'] + el['height'] <= 640, f"{el['element']} exceeds bottom edge"

# Check no coordinates in Stage 1 plan
import re
coord_match = re.search(r'\d+px|\bleft:\s*\d+|\btop:\s*\d+', plan)
assert coord_match is None, f'Stage 1 plan contains coords: {coord_match.group()}'

print('\n' + '='*60)
print('  ALL VALIDATION CHECKS PASSED')
print('='*60)
