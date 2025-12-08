import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# ********** עקיפת הייבוא של 'langchain.schema' **********
from google.api_core.exceptions import GoogleAPICallError

# יצירת מחלקה Document חלופית, במקום לייבא מ-langchain.schema
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# --- הגדרות קריטיות ---
os.environ["GEMINI_API_KEY"] = "AIzaSyCvVrLUQl73FBRusyunvnb-gsm_hGUgeJU"
API_KEY = os.environ["GEMINI_API_KEY"] 
DB_INDEX_PATH = "faiss_index"

# --- טמפלייט הפרומפט (שאלון) ---
PROMPT_TEMPLATE = """השתמש בנתוני ההקשר (CONTEXT) הבאים כדי לענות על השאלה. 
אם אינך יודע את התשובה, אמור שאינך יודע, ואל תנסה להמציא תשובה.

CONTEXT:
{context}

שאלה: {question}

תשובה עברית ברורה:"""

# --- טעינת הזיכרון וה-LLM ---

try:
    print(f"--- טוען את מסד הנתונים הווקטורי מ: {DB_INDEX_PATH}...")
    
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=API_KEY
    )
    
    # טעינת מסד הנתונים (FAISS)
    # הערה: FAISS.load_local מצליח לטעון את הנתונים גם ללא המחלקה המקורית
    db = FAISS.load_local(DB_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
    
    # הגדרת מודל Gemini לתשובות (ה-LLM)
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
    
    print("--- ✅ הבוט מוכן. שאל שאלה על 'עץ האורן' או כל נושא אחר שקיים בזיכרון! ---")
    print("-" * 40)
    
except GoogleAPICallError as e:
    print(f"!!! שגיאת API: המפתח לא תקין או שאין הרשאות. {e}")
    exit()
except FileNotFoundError:
    print(f"!!! שגיאה: לא נמצאה התיקייה {DB_INDEX_PATH}. ודא שהרצת את create_db.py קודם. !!!")
    exit()
except Exception as e:
    print(f"!!! שגיאה כללית בהפעלת הבוט: {e}")
    exit()

# --- לולאת שאלות ותשובות ---

while True:
    query = input("שאלתך (או 'יציאה' לסיום): ")
    if query.lower() in ["יציאה", "exit"]:
        print("להתראות!")
        break
    
    if not query:
        continue

    try:
        print("--- הבוט חושב... ---")
        
        # 1. אחזור המסמכים הרלוונטיים (Retrieval)
        # עדיין משתמש ב-FAISS שאמור לעבוד
        docs = db.similarity_search(query, k=4) 
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # 2. יצירת הפרומפט המלא
        full_prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        
        # 3. קריאה ישירה למודל (Generation)
        result = llm.invoke(full_prompt)
        
        # הדפסת התשובה
        print("\n🤖 תשובת הבוט:")
        print(result) 
        print("-" * 40)

    except Exception as e:
        print(f"אירעה שגיאה במהלך הקריאה ל-Gemini: {e}")