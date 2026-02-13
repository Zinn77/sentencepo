import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface/transformers'
from transformers import AutoTokenizer, AutoModelForCausalLM

