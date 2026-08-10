from datasets import load_dataset

ds = load_dataset("koutch/intro_prog", "dublin_data")

print(ds["train"][0])