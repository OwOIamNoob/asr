from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-lao")
tokenizer.save_vocabulary("/work/hpc/potato/laos_vi/data/embedding", "lao_tts_vocab")