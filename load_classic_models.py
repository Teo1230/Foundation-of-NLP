import re
import joblib
import glob
import re
import string
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Load the Romanian spaCy model
nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 4000000  # Or any value greater than your maximum text length
# Step 2: Encode labels
labels=['16', '16', '16', '17', '17', '17', '17', '18', '18', '18', '18', '19', '19', '19', '19', '20', '20', '20', '21', '21']
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

def has_vowel(word):
    """Check if the word contains at least one vowel."""
    vowels = "aeiouăâî"
    return any(char in vowels for char in word)


def prep(text):
    # Lowercase the text
    text = text.lower()

    # Replace specified characters with Romanian diacritics
    text = text.replace("[", "ă")
    text = text.replace("{", "ă")
    text = text.replace("`", "â")
    text = text.replace("=", "ș")
    text = text.replace("\\", "ț")  # Ensure the backslash is handled correctly
    text = text.replace("]", "î")
    text = text.replace("}", "Î")
    text = text.replace("|", "Ț")
    text = text.replace("º", "ș")
    text = text.replace("§", "")
    text = text.replace("˘", "")

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Roman numerals (I to XII and beyond)
    roman_numeral_pattern = r'\b(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))\b'
    text = re.sub(roman_numeral_pattern, '', text)

    # Process the text with spaCy
    doc = nlp_ro(text)

    # Filter out unwanted tokens and lemmatize
    sanitized_tokens = []
    for token in doc:
        # Conditions to filter tokens
        if (token.is_alpha and not token.is_stop and
                len(token) > 1 and has_vowel(token.lemma_)):  # Check for vowel presence
            sanitized_tokens.append(token.lemma_)  # Use the lemma of the token

    # Join the sanitized tokens back into a single string
    sanitized_text = ' '.join(sanitized_tokens)

    return sanitized_text
# Sample text for prediction
sample_text = """ ce se petrece... Şi ne pomenim într-una din zile că părintele vine la şcoală şi ne aduce un scaun nou şi
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

sample_text="""
Cuvânt prealiminariu despre isvoarele istoriei romănilor.

Romănii au trebuință astăzi să se întemeieze în patriotism și în curaj, și să căștige statornicie în caracter. Aceste resultate credem că s-ar dobăndi, cănd ei ar avea o bună Istorie națională, și cănd aceasta ar fi îndestul răspăndită. Privind la acel șâr de veacuri în care părinții nostrii au trăit, și la chipul cu care ei s-au purtat în vieața lor soțială, noi am căuta să dobândim virtuțile lor, și să ne ferim de greșelile în care au picat. Am părăsi, prin urmare, acel duh de partidă și de ambiție mărșavă. Am scăpa de acele temeri de nimic, d’acele nădejdi deșarte. Am dobândi adevăratele princine, care trebue să ne povățuiască în vieața noastră soțială, ca să ne putem măntui. Istoria încă ne ar arăta că părinții nostri se aflară în vremi cu mult mai grele decât acelea în care ne aflăm noi acum […]
Hrisovu de la Ioan Leon V(oe)v(od), pentru isgonirea Grecilor din țară. 1631.

Milostieio Bojieio [Din mila lui Dumnezeu] Ioan Leon Voevod și Domn al tot pământului Ungrovlahiei, fiiul marelui și prea bunului răposat Ioan Ștefan Voevod. Dat-am Domnia mea această carte a Domnii mele, ca să fie de mare credință tuturor popilor și diaconilor de pre orașe, și den toată țara Domnii mele de la toate sfintele biserici, câți popi și diaconi vor fi în par(o)fia a părintelui Episcopul Efrem de la sfânta Episcopie de la Buzău, ca să se știe pentru niște lucruri și obiceiuri rele ce au fost adaos de oameni streini în țara Domnii mele, care obicee nimenilea nu le-au mai putut obicni, căci văzând că sânt de mare pagubă țării. Pentru aceea Domnia mea am socotit de am strâns toată țara, boiari mari și mici, și roșii și mazâlii, și toți slujitorii de am sfătuit cu voia Domnii mele. Deci văzând toți atâta sărăcie și pustiire țării, căutat-am Domnia mea cu tot sfatul țării, să se afle de unde cad acele nevoi pre țară, aflatu-s-au și s-au adevărat cum toate nevoile și sărăcia țării se încep de la Greci streini, carii amestecă domniile și vând țara fără // milă, și o precunosc pre camete asuprite, și daca vin aici în țară ei nu socotesc să umble după obiceiul țării, ci strică toate lucrurile bune, și adaogă legi rele, și asuprite, și alte slujbe le au mărit și le au rădicat fără seamă pre atâta greime, ca să-și plătească ei cametele lor, și să-și îmbogățească casele lor, și încă alte multe streinări au arătat spre oameni țării, nesocotind pre nici un om de țară, înstreinând oamenii despre domnia mea cu pisme și cu năpăști, și asuprind săracii fără milă, și arătând vrăjmășie către toți oamenii țării. Deci văzând Domnia mea niște lucruri ca acelea și pradă țara, socotit-am Domnia mea dinpreună cu tot sfatul țării, de am făcut Domnia mea legătură și jurământ mare, de am jurat Domnia mea țării pre sfânta Evanghelie și cu mare afurisanie denaintea cinstitului și prea luminatului părintelui nostru al Țării Rumânești kir Vlădica Grigorie, și denaintea părintelui Teofil Episcopul de la Râmnic, în sfânta biserică a Domniei mele cea mare din cetatea scaunului Domniei mele din București, și după jurământ cu tot sfatul țării călcat-am acele obicee rele, și le am pus Domnia mea toate jos, și am scos acei Greci streini den țară afară ca pre niște nepreteni țării fiind, și am tocmit Domnia mea și alte lucruri bune care să fie de folos țării […]"""

sample_text = prep(sample_text)
# Load all models ending in `.joblib`
model_files = glob.glob("*.joblib")
models = {model_file: joblib.load(model_file) for model_file in model_files}
x = sorted(models)
# Predict the century with each model and print the results
print("Predictions:")
for model_name in x:
    model=models[model_name]
#for model_name, model in models.items():
    predicted_label = model.predict([sample_text])[0]

    predicted_century = label_encoder.inverse_transform([predicted_label])[0]

    print(f"{model_name} Prediction for given text: {predicted_century}\n")
