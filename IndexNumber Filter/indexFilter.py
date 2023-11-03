import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.preprocessing import text

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertForSequenceClassification.from_pretrained(model_name)

# Create a simple binary classification model
def create_model():
    input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.string)
    tokenized_input = tokenizer(input_layer['input_ids'], input_layer['attention_mask'], return_tensors='tf')
    output = bert_model(tokenized_input)['logits']
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example data
useful_strings = ["This is a useful string.", "Another useful example."]
not_useful_strings = ["This is not helpful at all.", "Ignore this one."]

# Prepare training data
train_data = useful_strings + not_useful_strings
train_labels = [1] * len(useful_strings) + [0] * len(not_useful_strings)

# Tokenize input data
tokenized_train_data = tokenizer(train_data, padding=True, truncation=True, return_tensors='tf')

# Train the model
model.fit(tokenized_train_data, train_labels, epochs=5)

# Test the model
test_strings = ["A new string to test.", "This is helpful."]
tokenized_test_data = tokenizer(test_strings, padding=True, truncation=True, return_tensors='tf')
predictions = model.predict(tokenized_test_data)
predicted_labels = tf.argmax(predictions['logits'], axis=1).numpy()

for string, label in zip(test_strings, predicted_labels):
    print(f"String: {string}\nPredicted Label: {'Useful' if label == 1 else 'Not Useful'}\n")
