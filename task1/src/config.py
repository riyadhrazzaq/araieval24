max_length = 256
lr = 6.189371832379065e-05  # from optuna
weight_decay = 0.0012404502272307953  # from optuna
warmup_steps = 120  # from optuna
batch_size = 32
model_name = "bert-base-multilingual-cased"
checkpoint_dir = "./checkpoints"
max_epoch = 5
random_seed = 111

valid_step_interval = 10
train_step_interval = 20

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

