import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *
# Load the saved model and tokenizer
output_dir = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(output_dir)

# Convert the century labels to numerical values for classification
century_to_label = {century: idx for idx, century in enumerate(sorted(os.listdir(os.path.join(os.getcwd(), 'century'))))}

# Function to preprocess and predict the century of the input text
def predict_century(input_text):
    # Preprocess the input text
    processed_text = prep(input_text)  # Use the same prep function as before

    # Tokenize the processed text
    encodings = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )

    # Get the model prediction
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert the predicted label back to century
    label_to_century = {idx: century for century, idx in century_to_label.items()}
    predicted_century = label_to_century[predictions.item()]

    return predicted_century

# Example input text
input_text = """ ce se petrece... Şi ne pomenim într-una din zile că părintele vine la şcoală şi ne aduce un scaun nou şi
    lung, şi
    după ce-a întrebat de dascăl, care cum ne purtăm, a stat puţin pe gânduri, apoi a pus nume scaunului -
    Calul Balan
    şi l-a lăsat în şcoală.
    În altă zi ne trezim că iar vine părintele la şcoală, cu moş Fotea, cojocarul satului, care ne aduce, dar
    de şcoală
    nouă, un drăguţ de biciuşor de curele, împletit frumos, şi părintele îi pune nume - Sfântul Nicolai,
    după cum este
    şi hramul bisericii din Humuleşti... Apoi pofteşte pe moş Fotea că, dacă i-or mai pica ceva curele
    bune, să mai facă
    aşa, din când în când, câte unul, şi ceva mai grosuţ, dacă se poate... Bădiţa Vasile a zâmbit atunci,
    iară noi, şcolarii,
    am rămas cu ochii holbaţi unii la alţii. Şi a pus părintele pravilă şi a zis că în toată sâmbăta să se
    procitească băieţii
    şi fetele, adică să asculte dascălul pe fiecare de tot ce-a învăţat peste săptămână; şi câte greşeli va
    face să i le
    însemne cu cărbune pe ceva, iar la urma urmelor, de fiecare greşeală să-i ardă şcolarului câte un
    sfânt-Nicolai.
    Atunci copila părintelui, cum era sprinţară şi plină de incuri, a bufnit în râs. Păcatul ei, sărmana!
"""

predicted_century = predict_century(input_text)
print(f"The predicted century for the input text is: {predicted_century}")
