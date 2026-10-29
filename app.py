from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import sqlite3
import re
import string
from functools import wraps
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from werkzeug.security import generate_password_hash, check_password_hash

# NLTK resources used by the NLP pipeline
for package, resource in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

app = Flask(__name__)
app.secret_key = "smart-upi-change-this-secret-key"

DB_NAME = "transactions.db"

CATEGORY_PROFILES = {
    "Food": "food restaurant swiggy zomato pizza burger dinner lunch breakfast cafe coffee grocery meal bakery snacks",
    "Shopping": "shopping amazon flipkart clothes fashion shoes electronics headphones mobile laptop purchase mall online order",
    "Travel": "travel uber ola taxi cab bus train flight metro petrol fuel parking ride transport ticket trip",
    "Bills": "bill electricity water gas internet mobile recharge phone rent insurance utility payment broadband",
    "Entertainment": "movie cinema netflix hotstar prime spotify music concert game gaming entertainment theatre ticket",
    "Others": "other transfer cash payment miscellaneous donation education medicine personal expense service"
}


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create/upgrade the existing database without deleting old transactions."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL,
            description TEXT,
            icon TEXT
        )
    """)

    cur.execute("PRAGMA table_info(transactions)")
    columns = [row[1] for row in cur.fetchall()]

    if not columns:
        cur.execute("""
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                description TEXT,
                amount REAL,
                date TEXT,
                merchant TEXT,
                category TEXT,
                confidence TEXT,
                cleaned_text TEXT,
                keywords TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
    else:
        # Upgrade the old table used by the original project.
        additions = {
            "user_id": "INTEGER",
            "confidence": "TEXT",
            "cleaned_text": "TEXT",
            "keywords": "TEXT"
        }
        for column, definition in additions.items():
            if column not in columns:
                cur.execute(f"ALTER TABLE transactions ADD COLUMN {column} {definition}")

    categories = [
        ("Food", "Restaurants, food delivery, groceries and meals", "fa-bowl-food"),
        ("Shopping", "Online and offline shopping purchases", "fa-cart-shopping"),
        ("Travel", "Taxi, fuel, public transport and travel", "fa-car"),
        ("Bills", "Utilities, recharge, rent and regular bills", "fa-file-invoice-dollar"),
        ("Entertainment", "Movies, music, games and entertainment", "fa-film"),
        ("Others", "Transactions that do not fit another category", "fa-box")
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO categories(category_name, description, icon) VALUES (?, ?, ?)",
        categories
    )

    # Create a demo account so the old transactions remain usable after migration.
    demo_email = "demo@smartupi.com"
    cur.execute("SELECT id FROM users WHERE email = ?", (demo_email,))
    demo = cur.fetchone()
    if not demo:
        cur.execute(
            "INSERT INTO users(name, email, password) VALUES (?, ?, ?)",
            ("Demo User", demo_email, generate_password_hash("demo123"))
        )
        demo_user_id = cur.lastrowid
    else:
        demo_user_id = demo["id"]

    # Existing transactions from the old version are assigned to the demo account.
    cur.execute("UPDATE transactions SET user_id = ? WHERE user_id IS NULL", (demo_user_id,))

    conn.commit()
    conn.close()


init_db()


def login_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("home"))
        return function(*args, **kwargs)
    return decorated_function


def preprocess_text(text):
    """Advanced NLP preprocessing: lowercase, tokenization, stopword removal and lemmatization."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    lemmatizer = WordNetLemmatizer()
    cleaned = []
    for token in tokens:
        if token in stop_words or token in string.punctuation or len(token) < 2:
            continue
        try:
            token = lemmatizer.lemmatize(token)
        except LookupError:
            pass
        cleaned.append(token)

    return tokens, cleaned


def predict_category(text):
    """TF-IDF + cosine similarity classification with a rule-based fallback."""
    _, cleaned = preprocess_text(text)
    cleaned_text = " ".join(cleaned)

    if SKLEARN_AVAILABLE and cleaned_text.strip():
        categories = list(CATEGORY_PROFILES.keys())
        documents = [CATEGORY_PROFILES[c] for c in categories]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents + [cleaned_text])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        best_index = similarities.argmax()
        score = float(similarities[best_index])

        # Give direct merchant/category clues a small boost.
        direct_words = {
            "swiggy": "Food", "zomato": "Food", "pizza": "Food",
            "amazon": "Shopping", "flipkart": "Shopping",
            "uber": "Travel", "ola": "Travel",
            "electricity": "Bills", "recharge": "Bills",
            "netflix": "Entertainment", "movie": "Entertainment"
        }
        for word, category in direct_words.items():
            if word in cleaned_text:
                if category == categories[best_index]:
                    score = max(score, 0.90)
                else:
                    best_index = categories.index(category)
                    score = max(score, 0.90)
                break

        confidence = min(99.0, max(60.0, 60.0 + score * 39.0))
        return categories[best_index], f"{confidence:.1f}%", cleaned_text, cleaned

    # Fallback if scikit-learn is unavailable.
    rules = [
        ("Food", ["swiggy", "zomato", "pizza", "food", "restaurant", "burger", "dinner"]),
        ("Shopping", ["amazon", "flipkart", "shopping", "clothes", "shoes", "headphones"]),
        ("Travel", ["uber", "ola", "taxi", "fuel", "petrol", "train", "flight"]),
        ("Bills", ["electricity", "bill", "recharge", "rent", "internet"]),
        ("Entertainment", ["movie", "netflix", "spotify", "game", "cinema"])
    ]
    for category, words in rules:
        if any(word in cleaned_text for word in words):
            return category, "90.0%", cleaned_text, cleaned
    return "Others", "65.0%", cleaned_text, cleaned


@app.route("/", methods=["GET"])
def home():
    # The existing index.html acts as the login/register page when logged out,
    # and as the personalized home page after login.
    if "user_id" not in session:
        mode = request.args.get("mode", "login")
        return render_template("index.html", auth_mode=mode)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM transactions WHERE user_id = ? ORDER BY id DESC LIMIT 3",
        (session["user_id"],)
    )
    recent = cur.fetchall()
    conn.close()
    return render_template("index.html", recent_transactions=recent, user_name=session.get("user_name"))


@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not name or not email or not password:
        return render_template("index.html", auth_mode="register", error="Please fill all fields.")

    if len(password) < 6:
        return render_template("index.html", auth_mode="register", error="Password must contain at least 6 characters.")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users(name, email, password) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return render_template("index.html", auth_mode="register", error="This email is already registered.")
    conn.close()
    return redirect(url_for("home"))


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user and check_password_hash(user["password"], password):
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        return redirect(url_for("home"))

    return render_template("index.html", auth_mode="login", error="Invalid email or password.")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    description = request.form.get("description", "").strip()
    amount = request.form.get("amount", "0")
    date = request.form.get("date", "")
    merchant = request.form.get("merchant", "").strip()

    if not description or not amount or not date:
        return render_template("analyze.html", error="Please enter description, amount and date.")

    try:
        amount_value = float(amount)
        if amount_value <= 0:
            raise ValueError
    except ValueError:
        return render_template("analyze.html", error="Please enter a valid amount.")

    tokens, cleaned_tokens = preprocess_text(description)
    category, confidence, cleaned_text, keywords = predict_category(description)

    if not merchant:
        merchant = "Auto-detected from description"

    insight = (
        f"The NLP engine classified this transaction as {category} with {confidence} confidence. "
        f"Important terms identified: {', '.join(keywords[:6]) if keywords else 'none'}."
    )

    conn = get_db()
    conn.execute("""
        INSERT INTO transactions
        (user_id, description, amount, date, merchant, category, confidence, cleaned_text, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"], description, amount_value, date, merchant,
        category, confidence, cleaned_text, ", ".join(keywords)
    ))
    conn.commit()
    conn.close()

    return render_template(
        "analyze.html",
        original_text=description,
        cleaned_text=cleaned_text,
        tokens=tokens,
        keywords=keywords,
        category=category,
        confidence=confidence,
        merchant=merchant,
        amount=amount_value,
        insight=insight
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Personalized dashboard with month/year/custom transaction filtering."""
    conn = get_db()
    cur = conn.cursor()
    uid = session["user_id"]

    # --------------------------
    # FILTERS
    # --------------------------
    period = request.args.get("period", "all")
    month = request.args.get("month", "")
    year = request.args.get("year", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    category_filter = request.args.get("category", "")

    where = ["user_id = ?"]
    params = [uid]

    # Month-wise filtering: YYYY-MM
    if period == "month" and month:
        where.append("substr(date, 1, 7) = ?")
        params.append(month)

    # Year-wise filtering: YYYY
    elif period == "year" and year:
        where.append("substr(date, 1, 4) = ?")
        params.append(year)

    # Custom date range
    elif period == "custom":
        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)

    if category_filter:
        where.append("category = ?")
        params.append(category_filter)

    where_sql = " AND ".join(where)

    # Filtered transactions
    cur.execute(
        f"SELECT * FROM transactions WHERE {where_sql} ORDER BY date DESC, id DESC",
        params
    )
    transactions = cur.fetchall()

    # Total expense
    cur.execute(
        f"SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE {where_sql}",
        params
    )
    total_expense = cur.fetchone()[0]

    # Total transactions
    cur.execute(
        f"SELECT COUNT(*) FROM transactions WHERE {where_sql}",
        params
    )
    total_transactions = cur.fetchone()[0]

    # Top category
    cur.execute(f"""
        SELECT category, SUM(amount) AS total_amount
        FROM transactions
        WHERE {where_sql}
        GROUP BY category
        ORDER BY total_amount DESC
        LIMIT 1
    """, params)
    result = cur.fetchone()
    top_category = result["category"] if result else "No Data"

    # Category chart data
    cur.execute(f"""
        SELECT category, SUM(amount) AS total
        FROM transactions
        WHERE {where_sql}
        GROUP BY category
        ORDER BY total DESC
    """, params)
    chart_data = cur.fetchall()
    labels = [row["category"] for row in chart_data]
    values = [row["total"] for row in chart_data]

    average_expense = total_expense / total_transactions if total_transactions else 0
    progress_data = []
    for row in chart_data:
        percentage = (row["total"] / total_expense * 100) if total_expense else 0
        progress_data.append({
            "category": row["category"],
            "percentage": round(percentage, 1)
        })

    if total_transactions:
        insight = (
            f"Your highest spending category is {top_category}. "
            f"You have {total_transactions} transaction(s), with an average transaction value "
            f"of ₹{average_expense:.2f} for the selected period."
        )
    else:
        insight = "No transactions found for the selected period. Try another month, year, or date range."

    # Years available for the year dropdown
    cur.execute(
        "SELECT DISTINCT substr(date, 1, 4) AS year FROM transactions "
        "WHERE user_id = ? AND date IS NOT NULL AND date != '' ORDER BY year DESC",
        (uid,)
    )
    available_years = [row["year"] for row in cur.fetchall() if row["year"]]

    conn.close()

    return render_template(
        "dashboard.html",
        transactions=transactions,
        total_expense=total_expense,
        total_transactions=total_transactions,
        top_category=top_category,
        labels=labels,
        values=values,
        average_expense=average_expense,
        progress_data=progress_data,
        insight=insight,
        user_name=session.get("user_name"),
        period=period,
        month=month,
        year=year,
        start_date=start_date,
        end_date=end_date,
        category_filter=category_filter,
        available_years=available_years,
        categories=["Food", "Shopping", "Travel", "Bills", "Entertainment", "Others"]
    )


@app.route("/download_report")
@login_required
def download_report():
    """Generate a filtered PDF expense statement useful for bookkeeping and GST preparation."""
    conn = get_db()
    cur = conn.cursor()
    uid = session["user_id"]

    period = request.args.get("period", "all")
    month = request.args.get("month", "")
    year = request.args.get("year", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    category_filter = request.args.get("category", "")

    where = ["user_id = ?"]
    params = [uid]

    if period == "month" and month:
        where.append("substr(date, 1, 7) = ?")
        params.append(month)
    elif period == "year" and year:
        where.append("substr(date, 1, 4) = ?")
        params.append(year)
    elif period == "custom":
        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)

    if category_filter:
        where.append("category = ?")
        params.append(category_filter)

    where_sql = " AND ".join(where)

    cur.execute(
        f"SELECT * FROM transactions WHERE {where_sql} ORDER BY date ASC, id ASC",
        params
    )
    transactions = cur.fetchall()

    cur.execute(
        f"SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE {where_sql}",
        params
    )
    total_expense = cur.fetchone()[0]

    cur.execute(
        f"SELECT COUNT(*) FROM transactions WHERE {where_sql}",
        params
    )
    total_transactions = cur.fetchone()[0]

    cur.execute(f"""
        SELECT category, COUNT(*) AS transaction_count, SUM(amount) AS total_amount
        FROM transactions
        WHERE {where_sql}
        GROUP BY category
        ORDER BY total_amount DESC
    """, params)
    category_summary = cur.fetchall()

    user = conn.execute("SELECT name, email FROM users WHERE id = ?", (uid,)).fetchone()
    conn.close()

    # Human-readable report period
    if period == "month" and month:
        try:
            report_period = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
        except ValueError:
            report_period = month
    elif period == "year" and year:
        report_period = f"Year {year}"
    elif period == "custom":
        report_period = f"{start_date or 'Beginning'} to {end_date or 'Present'}"
    else:
        report_period = "All Transactions"

    if category_filter:
        report_period += f" | Category: {category_filter}"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Smart UPI Expense Statement"
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=18, leading=22, spaceAfter=6
    )
    sub_style = ParagraphStyle(
        "ReportSub", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=9, textColor=colors.grey, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontSize=12,
        leading=15, spaceBefore=8, spaceAfter=6
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=7.5, leading=9
    )

    story = []
    story.append(Paragraph("Smart UPI Expense Statement", title_style))
    story.append(Paragraph("Transaction summary for bookkeeping and GST/tax record preparation", sub_style))

    customer_name = user["name"] if user else session.get("user_name", "User")
    customer_email = user["email"] if user else ""

    info_data = [
        ["Account Holder", customer_name, "Email", customer_email],
        ["Report Period", report_period, "Generated On", datetime.now().strftime("%d-%m-%Y %H:%M")],
        ["Total Transactions", str(total_transactions), "Total Expense", f"Rs. {total_expense:,.2f}"]
    ]
    info_table = Table(info_data, colWidths=[35*mm, 75*mm, 35*mm, 75*mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0efff")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#f0efff")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Category-wise Expense Summary", heading_style))
    summary_data = [["Category", "Transactions", "Total Expense (Rs.)", "Share"]]
    for row in category_summary:
        share = (row["total_amount"] / total_expense * 100) if total_expense else 0
        summary_data.append([
            row["category"] or "Others",
            str(row["transaction_count"]),
            f"{row['total_amount']:,.2f}",
            f"{share:.1f}%"
        ])
    if len(summary_data) == 1:
        summary_data.append(["No data", "0", "0.00", "0.0%"])

    summary_table = Table(summary_data, colWidths=[55*mm, 35*mm, 55*mm, 35*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6c63ff")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8ff")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Transaction Register", heading_style))
    transaction_data = [["ID", "Date", "Merchant", "Description", "Category", "Amount (Rs.)", "NLP Confidence"]]
    for row in transactions:
        description = (row["description"] or "")[:65]
        merchant = (row["merchant"] or "")[:28]
        transaction_data.append([
            str(row["id"]),
            str(row["date"] or ""),
            Paragraph(merchant, small_style),
            Paragraph(description, small_style),
            str(row["category"] or "Others"),
            f"{row['amount']:,.2f}",
            str(row["confidence"] or "N/A")
        ])
    if len(transaction_data) == 1:
        transaction_data.append(["-", "-", "-", "No transactions found for this period", "-", "0.00", "-"])

    transaction_table = Table(
        transaction_data,
        repeatRows=1,
        colWidths=[13*mm, 25*mm, 42*mm, 72*mm, 32*mm, 30*mm, 32*mm]
    )
    transaction_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6c63ff")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.lightgrey),
        ("FONTSIZE", (0,0), (-1,-1), 7.2),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafaff")]),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(transaction_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Note: This report is a transaction/expense record generated from the Smart UPI database. "
        "It can support bookkeeping, expense review and GST/tax preparation, but it is not itself an official GST return or tax invoice. "
        "GST-specific fields such as GSTIN, tax rate, CGST, SGST and IGST are not currently stored in this application.",
        small_style
    ))

    doc.build(story)
    buffer.seek(0)

    safe_period = re.sub(r"[^A-Za-z0-9_-]+", "_", report_period)[:50]
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Smart_UPI_Expense_Report_{safe_period}.pdf",
        mimetype="application/pdf"
    )


@app.route("/ExpenseCategories")
@login_required
def categories():
    conn = get_db()
    cur = conn.cursor()
    uid = session["user_id"]

    data = {}
    for category in ["Food", "Shopping", "Travel", "Bills", "Entertainment", "Others"]:
        cur.execute(
            "SELECT * FROM transactions WHERE user_id = ? AND category = ? ORDER BY id DESC",
            (uid, category)
        )
        data[category.lower()] = cur.fetchall()

    # Full database view for the logged-in user.
    cur.execute("SELECT * FROM transactions WHERE user_id = ? ORDER BY id DESC", (uid,))
    all_transactions = cur.fetchall()

    conn.close()
    return render_template("ExpenseCategories.html", **data, all_transactions=all_transactions,
                           user_name=session.get("user_name"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
