import os
from langchain_google_genai import GoogleGenerativeAI
from google.api_core.exceptions import GoogleAPICallError

# 1. הגדרת המפתח הסודי של Gemini (ודא שזה המפתח שלך ובתוך גרשיים)
os.environ["GEMINI_API_KEY"] = "AIzaSyCvVrLUQl73FBRusyunvnb-gsm_hGUgeJU" 

# 2. בדיקת חיבור באמצעות LangChain
try:
    print("--- מנסה להתחבר ל-Gemini דרך LangChain... ---")
    
    # ניסיון ליצור אובייקט של Gemini
    llm = GoogleGenerativeAI(model="gemini-2.5-flash")
    
    # ניסיון לבצע קריאה פשוטה ל-API (פעולה זו מוודאת שהמפתח עובד)
    test_query = "שלום, האם אתה עובד?"
    response = llm.invoke(test_query)
    
    print("\n--- ✅ הצלחה! החיבור ל-Gemini עובד דרך LangChain! ✅ ---")
    print(f"קיבלת תשובה לדוגמה: {response[:40]}...") 
except GoogleAPICallError as e:
    print("\n!!! ❌ שגיאת API: המפתח לא עובד או שהרשאות חסרות. אנא ודא שהמפתח תקין. !!!")
except Exception as e:
    # אם יש שגיאת ייבוא, זה יתפוס אותה
    print(f"\n!!! ❌ שגיאה כללית: {e}")