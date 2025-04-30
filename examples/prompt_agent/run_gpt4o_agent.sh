#!/usr/bin/env bash

# Usage:
#   Set ENV_NAME to "webshop" or "alfoworld"
#   Then run this script to launch the corresponding agent.

ENV_NAME="webshop"   # webshop or alfoworld

if [[ "$ENV_NAME" == "webshop" ]]; then
  echo "Launching Webshop agent..."
  python3 -m examples.prompt_agent.gpt4o_webshop
elif [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_alfworld
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'webshop' or 'alfoworld'." >&2
  exit 1
fi
