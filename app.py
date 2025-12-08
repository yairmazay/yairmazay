import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.api_core.exceptions import GoogleAPICallError

# יצירת מחלקה Document חלופית (כדי לעקוף את שגיאת הייבוא)
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

# --- אתחול ה-API והמודלים (מתבצע פעם אחת בעת הפעלת השרת) ---

app = Flask(__name__)
# מאפשר קריאות Cross-Origin (מהאתר שלך לשרת ה-API)
CORS(app) 

db = None
llm = None

def init_bot():
    """טוען את מסד הנתונים ואת המודל."""
    global db, llm
    try:
        print("--- מתחיל לטעון את הבוט לשרת ה-API... ---")
        
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=API_KEY
        )
        
        db = FAISS.load_local(DB_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
        
        print("--- ✅ הבוט מוכן ומחכה לבקשות API. ---")
        return True
    
    except FileNotFoundError:
        print(f"!!! שגיאה: לא נמצאה התיקייה {DB_INDEX_PATH}. ודא שהרצת את create_db.py קודם. !!!")
        return False
    except GoogleAPICallError as e:
        print(f"!!! שגיאת API: המפתח לא תקין או שאין הרשאות. {e}")
        return False
    except Exception as e:
        print(f"!!! שגיאה כללית באתחול הבוט: {e}")
        return False

# --- הגדרת נקודת הקצה (Endpoint) של ה-API ---

@app.route('/ask', methods=['POST'])
def ask_bot():
    """מקבל שאילתה ומחזיר תשובה מהמודל."""
    if not db or not llm:
        return jsonify({"error": "Bot not initialized. Check server logs."}), 500

    # קבלת הנתונים (השאלה) מהבקשה
    data = request.get_json()
    query = data.get('question', '')

    if not query:
        return jsonify({"error": "לא נשלחה שאלה."}), 400

    try:
        # 1. אחזור המסמכים הרלוונטיים (Retrieval)
        docs = db.similarity_search(query, k=4) 
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # 2. יצירת הפרומפט המלא
        full_prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        
        # 3. קריאה ישירה למודל (Generation)
        result = llm.invoke(full_prompt)
        
        # החזרת התשובה כ-JSON
        return jsonify({"answer": result, "status": "success"})

    except Exception as e:
        print(f"שגיאה במהלך הקריאה ל-Gemini: {e}")
        return jsonify({"error": "שגיאה פנימית בשרת ה-API."}), 500

# --- הפעלת השרת ---

if __name__ == '__main__':
    if init_bot():
        # השרת ירוץ על פורט 5000 (ברירת המחדל של Flask)
        app.run(debug=True)