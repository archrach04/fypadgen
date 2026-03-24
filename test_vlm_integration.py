# Test VLM/LM integration
print('Testing VLM/LM Integration...')
print()

# Import app module to trigger wiring
import app
print(f'GEMINI_ENABLED (GitHub Models): {app.GEMINI_ENABLED}')
print(f'LLM Model: {app.GITHUB_LLM_MODEL}')
print(f'VLM Model: {app.GITHUB_VLM_MODEL}')
print(f'Endpoint: {app.GITHUB_MODELS_ENDPOINT}')
print()

# Check if functions are wired
from laygen_pricemapping import VLM_ANALYZE_FUNC, LM_RERANK_FUNC
print(f'VLM_ANALYZE_FUNC set: {VLM_ANALYZE_FUNC is not None}')
print(f'LM_RERANK_FUNC set: {LM_RERANK_FUNC is not None}')
print()

# Test VLM analysis (with a sample product image URL)
test_url = 'https://m.media-amazon.com/images/I/71GLMJ2E27L._SX679_.jpg'
print(f'Testing VLM analysis with: {test_url[:50]}...')
result = app.analyze_image_with_vlm(test_url, 'layout')
print(f'VLM Result success: {result.get("success", False)}')
if result.get('analysis'):
    print(f'Analysis keys: {list(result["analysis"].keys())[:5]}')
elif result.get('raw'):
    raw = result['raw'][:300] if len(result.get('raw', '')) > 300 else result.get('raw', '')
    print(f'Raw response: {raw}')
elif result.get('error'):
    print(f'Error: {result.get("error")}')
else:
    print(f'Result: {result}')
print()
print('Integration test complete!')
