from utils import *
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Step 1: Load the model and tokenizer
output_dir = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)

# Step 2: Load the label encoder
label_encoder = joblib.load('./label_encoder.pkl')  # Load the saved label encoder

# Step 3: Function to predict the century of the input text
def predict_century(input_text):
    # Preprocess the input text
    encoding = tokenizer(
        input_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

    # Make sure to put the model in evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**encoding)

    # Get the predicted class
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_label = predictions.item()

    # Decode the predicted label back to the original century
    predicted_century = label_encoder.inverse_transform([predicted_label])
    return predicted_century[0]

# Step 4: Test the prediction with an example input text
# Example input text
input_text= """
astfel răsturnat... Continuaţi."
Hegel răsturnat! Nu rîdeam cu gura pînă la urechi, cum făcea Matilda, dar
rîdeam foarte tare în forul meu interior, văzînd cum un dentist îi amenda pe
marii filozofi luînd drept literă de evanghelie ceea ce un geniu politic cum
fusese Lenin spusese, pentru necesităţi dictate de lupta ideologică pentru
cucerirea puterii, în cutare sau cutare împrejurare. Dentistul le scotea
filozofilor dinţii, îi plomba şi în cele din urmă le muta şi fălcile fără măcar să
clipească. Era foarte sigur de sine, dacă nu mai mult, şi asta era cu atît mai
curios cu cît îl ştiam pasionat de marile sisteme de la Platon şi Aristotel, pînă la
ontologia existenţială a lui Martin Heidegger. Această stranie şi rapidă
transformare, care îl făcea în primul rînd antipatic, constituia pentru mine un
mister. Ce i se întîmplase? Şi cînd? Cum îşi înăbuşise el ardoarea gîndirii sale
libere, care mă făcea să-mi amintesc adesea de el cu acea bucurie pe care ne-o
dă ideea că undeva în oraşul în care trăim cineva are aceleaşi preocupări ca şi
noi, deşi meseria lui e alta şi că oricînd te puteai duce să-l vezi, şi el să te primească cu acea sticlire în ochi pe care i-o dădea pasiunea nobilă pentru
cultură? Cum de renunţase şi la meseria lui şi la această pasiune şi acceptase
să vină la noi şi să... Abia acasă deschisei şi eu fălcile şi rîsei cu gura pînă la
urechi de mai multe ori: ha, ha, ha... ha, ha, ha, ha... ha, ha, ha, ha... "Ei, ce e,
mă întrebă Matilda. Ce te-a apucat?" "Avem un nou decan, îi spusei, un
dentist, ha, ha, ha, ha, ha... hi, hi, hi, hi, hi... N-are dreptate Marx, nu ne
despărţim numai de trecut rîzînd, ci intrăm şi în viitor tot rîzînd... însă de noi
înşine..." Şi îi povestii întîlnirea mea pe coridoare cu doctorul Vaintrub şi cum îi
spusesem că stau bine cu dantura, pot să sparg şi sîmburi de măsline... "Se
pregăteşte o reformă a învăţămîntului, continuai în timp ce mîncam flămînd.
Ce-ar putea să aducă nou această reformă. Ce putea să aducă? Elevul şi
studentul trebuiau oricum să înveţe? Bineînţeles! Acest adevăr nu putea fi
reformat. Atunci?!" "Cine ţi-a spus că acest adevăr nu poate fi reformat? rîse
Matilda mai tare parcă decît mine. Şeful catedrei voastre de filozofie a culturii
era prezent?" "Nu, lipseşte de vreo două săptămîni, are o sciatică infecţioasă,
care îl ţine la pat." "Hm! Aşa credeţi voi! De sciatica de care suferă el n-o să se
mai vindece niciodată. Marele vostru filozof a fost întîmpinat într-o zi la poarta
universităţii de către... nu-ţi spun cine şi i s-a spus: din clipa asta
dumneavoastră n-aveţi ce mai căuta aici." Mă posomorii, avînd într-o secundă
sentimentul că pentru mine perioada Matilda fusese foarte scurtă. Cum,
marelui poet şi filozof i se luase catedra? "Chiar e adevărat?!" spusei.
"Bineînţeles! zise Matilda care continua să fie veselă, ieri am aflat, dar am uitat
să-ţi spun."
Nu-l iubeam pe marele poet nici ca filozof, nici ca profesor, deşi el mă
susţinuse să fiu numit la catedra lui. Nu-mi plăcuse însă că era distant cu
mine şi se complăcea să fie înconjurat de adulatori găunoşi, care în afară de
opera lui, din care îi citau chiar în faţă fără nici o jenă, ştiau prea puţină
filozofie. Era adevărat că în sinea mea consideram lipsită de interes pentru
timpurile moderne teoria sa asupra unei cunoaşteri limitate de o cenzură
transcendentă, precum şi adaptarea la spaţiul românesc a gîndirii spengleriene
conform căreia culturile şi civilizaţiile sînt determinate de spaţiul geografic, dar 

"""
input_text = prep(input_text)
print(input_text)
predicted_century = predict_century(input_text)
print(f"Predicted Century: {predicted_century}")
