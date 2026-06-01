#!/usr/bin/env python3
"""Fix token extraction in telegram_ingestor.py"""
import re

with open('/opt/data/.env') as f:
    env = f.read()

m = re.search(r'TELEGRAM_BOT_TOKEN=*** env)
if not m:
    print("ERROR: token not found")
    exit(1)

token = m.group(1)
print(f"Token: {token[:10]}...{token[-5:]} ({len(token)} chars)")

with open('/opt/code/neuroflow-core/telegram_ingestor.py') as f:
    content = f.read()

# Replace the broken token extraction block
old_block = """TELEGRAM_TOKEN=***    open(\"/opt/data/.env\").read().split(\"TELEGRAM_BOT_TOKEN=\")[1]...0]
    if \"TELEGRAM_BOT_TOKEN=*** in open(\"/opt/data/.env\").read()
    else \"\"
)

BOT_TOKEN=***\"\"\"""""

new_block = f'''BOT_TOKEN = "{token}"'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Fixed broken token block")
else:
    print("Old block not found, trying alternative...")
    # Find any line with broken token extraction
    for i, line in enumerate(content.split('\n')):
        if 'TELEGRAM_TOKEN=' in line or 'BOT_TOKEN=*** in line:
            print(f"Line {i}: {line[:80]}")

with open('/opt/code/neuroflow-core/telegram_ingestor.py', 'w') as f:
    f.write(content)
print("Done")
