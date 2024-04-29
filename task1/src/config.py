max_length = 256
lr = 0.0001
epochs = 10
batch_size = 8
weight_decay = 0.
labels = [
    "Appeal_to_Values",
    "Loaded_Language",
    "Consequential_Oversimplification",
    "Causal_Oversimplification",
    "Questioning_the_Reputation",
    "Straw_Man",
    "Repetition",
    "Guilt_by_Association",
    "Appeal_to_Hypocrisy",
    "Conversation_Killer",
    "False_Dilemma-No_Choice",
    "Whataboutism",
    "Slogans",
    "Obfuscation-Vagueness-Confusion",
    "Name_Calling-Labeling",
    "Flag_Waving",
    "Doubt",
    "Appeal_to_Fear-Prejudice",
    "Exaggeration-Minimisation",
    "Red_Herring",
    "Appeal_to_Popularity",
    "Appeal_to_Authority",
    "Appeal_to_Time",
]

model_name = "bert-base-multilingual-cased"
checkpoint_dir = "./checkpoints"
max_epoch = 10
