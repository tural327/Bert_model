# Repository Contain Various type of Bert models 

- [x] [Sentiment Analysis](https://huggingface.co/blog/sentiment-analysis-python) - [My Model](https://github.com/tural327/Bert_model/blob/main/Untitled8.ipynb)
- [ ] [Question Answering](https://huggingface.co/docs/transformers/tasks/question_answering)

> Sentiment Analysis

I used Kaggel data for basic text cleaning I used :

```python
def text_preprocessing(df,col_name):
    column = col_name
    df[column] = df[column].progress_apply(lambda x:str(x).lower())
    df[column] = df[column].progress_apply(lambda x: th.cont_exp(x)) #you're -> you are; i'm -> i am
    df[column] = df[column].progress_apply(lambda x: th.remove_emails(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_html_tags(x))
    df[column] = df[column].progress_apply(lambda x: ps.remove_stopwords(x))

    df[column] = df[column].progress_apply(lambda x: th.remove_special_chars(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_accented_chars(x))
    df[column] = df[column].progress_apply(lambda x: th.make_base(x)) #ran -> run,
    return(df)
```
I build simple text classification model :
```python
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
out = tf.keras.layers.Dense(128, activation='relu')(outputs['pooled_output'])
out = tf.keras.layers.Dropout(0.1, name="dropout")(out)
l = tf.keras.layers.Dense(4, activation='sigmoid', name="output")(out)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])
```
First Part of training was not so well

| Class Name  | Precision | Recall  | f1-score |
| -------- | ------------- | ------------- | ------------- |
| 0  | 0.51  | 0.81  | 0.63  |
| 1  | 0.57  | 0.54  | 0.56  |
| 2  | 0.49  | 0.48  | 0.49  |
| 3  | 0.52  | 0.05  | 0.10  |
 
accuracy - 0.52
related with imbalance data

For fixing(simple approch) I droped 3 class because it was too low quantity 
and make second train

| Class Name  | Precision | Recall  | f1-score |
| -------- | ------------- | ------------- | ------------- |
| 0  | 0.62  | 0.76  | 0.68  |
| 1  | 0.64  | 0.64  | 0.64  |
| 2  | 0.64  | 0.49  | 0.55  |

accuracy - 0.63 

Model improved but still not best need more cleaning our dataset but I just tried to improve NLP skills
