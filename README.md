## This GitHub-Repo contains the code, sources and findings of the final project in the IANNwTF course
For the final project we built a small LSTM language model, trained it on the wiki_auto dataset. After training we evaluate it for intersectionality biases, specifically the crossing of gender and race. Following some ideas of the papers we used, we prompted the model with a combination of race, gender, profession. E.g. "This Asian female doctor is very". And counted the adjectives related to the prompted text, classified the adjectives into "negative, neutral and positive" connotation and and evaluated it.

We planned to evaluate our LSTM-model as well, but, as written in the paper, the sentences were incoherent and lacked adjectives.
The report can be found here: [report](report.pdf)

## How to use the [generator.py](NLP-Wikidataset/generator.py)
Select a model you want to use. Reference for individual model performance [here](NLP-Wikidataset/7_models.png).
You can find the models under [NLP-Wikidataset/model/LSTM/](NLP-Wikidataset/model/LSTM/).
Edit line 16 with the correct model name and run the file.
The outputs will be written to [predictions.txt](NLP-Wikidataset/predictions.txt). From there on you can analyse it by hand, as proposed in the report 

