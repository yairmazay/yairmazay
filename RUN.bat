@echo off
REM *** שלב 1: הגדרת המפתח הסודי של Gemini ***
set GEMINI_API_KEY=gen-lang-client-0861793642

REM *** שלב 2: הרצת קובץ יצירת הזיכרון ***
echo מנסה ליצור את הזיכרון החכם...
python create_db.py

if %errorlevel% neq 0 (
    echo !!! שגיאה !!! הזיכרון לא נוצר.
    echo וודא שהקבצים create_db.py ו-etz_haoren_data.txt נמצאים בתיקייה זו.
) else (
    echo ====== הצלחה! הזיכרון נוצר. ======
)

pause