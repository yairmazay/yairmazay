import os
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# 1. הגדרת המפתח הסודי (פתרון סופי ל-DefaultCredentialsError)
# א. הגדרת המפתח בסביבה
os.environ["GEMINI_API_KEY"] = "AIzaSyCvVrLUQl73FBRusyunvnb-gsm_hGUgeJU"
# ב. יצירת משתנה מקומי לשימוש ישיר במודל
API_KEY = os.environ["GEMINI_API_KEY"] 

# 2. שם קובץ הנתונים (משתמש בשם הקובץ המקורי כפי שנראה בתיקייה)
# ודא שהקובץ נמצא בתיקייה AI_BOT וששמו המדויק הוא etz_haoren_data.txt
DATA_FILE = "etz_haoren_data.txt" 

# 3. בדיקה לפני טעינה (מטפלת ב-FileNotFoundError)
if not os.path.exists(DATA_FILE):
    # אם הקובץ לא נמצא, נציג שגיאה ברורה
    raise FileNotFoundError(f"!!! שגיאה קריטית: הקובץ '{DATA_FILE}' לא נמצא. ודא ששם הקובץ בתיקייה AI_BOT מדויק לחלוטין. !!!")


# 4. טעינת המידע מקובץ הטקסט
print("--- טוען נתונים מקובץ... ---")
# ה-encoding="utf-8" מטפל בבעיות קריאת עברית
loader = TextLoader(DATA_FILE, encoding="utf-8")
documents = loader.load()


# 5. חלוקת המסמך לחלקים קטנים (Chunks)
print("--- מחלק את הטקסט לחלקים... ---")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# 6. הכנת מודל ה-Embedding (פתרון סופי ל-DefaultCredentialsError ע"י העברת המפתח ישירות)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=API_KEY
)


# 7. יצירת מסד הנתונים הווקטורי (הזיכרון)
print("--- יוצר זיכרון וקטורי (FAISS). אנא המתן... ---")
db = FAISS.from_documents(texts, embeddings_model)


# 8. שמירת הזיכרון לשימוש עתידי
db.save_local("faiss_index") 
print("--- הצלחה! הזיכרון נוצר בתיקייה faiss_index. ---")