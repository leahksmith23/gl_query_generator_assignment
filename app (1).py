"""
Northbridge Bank — Credit Risk Query Engine
Streamlit application

Converts the notebook-based PoC (verified query library + LLM-driven SQL
generation / validation / retry / narrative pipeline) into an interactive
Streamlit app for business users and analysts.
"""

from typing import Optional
import json
import os
import re
import sqlite3

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine | Northbridge Bank",
    page_icon="🏦",
    layout="wide",
)

DB_PATH = "credit_risk_portfolio.db"
TEST_QUERIES_PATH = "test_queries.csv"  # optional, used only for the batch-evaluation tab

# --------------------------------------------------------------------------
# Credentials / configuration
# --------------------------------------------------------------------------
# Preference order:
#   1. Streamlit secrets (st.secrets)      -> recommended for deployed apps
#   2. Environment variables               -> OPENAI_API_KEY / OPENAI_API_BASE
#   3. config.json in the working dir      -> same shape used in the notebook
#   4. Manual entry in the sidebar (session-only, not persisted)
def load_credentials():
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        api_base = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    # 2. Environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE")

    # 3. config.json (same format as the notebook)
    if (not api_key) and os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    return api_key, api_base


if "openai_api_key" not in st.session_state:
    key, base = load_credentials()
    st.session_state["openai_api_key"] = key
    st.session_state["openai_api_base"] = base


# --------------------------------------------------------------------------
# Sidebar — configuration & credentials
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("openai_api_key") or "",
        type="password",
        help="Loaded automatically from Streamlit secrets, environment variables, "
             "or config.json if available. Otherwise, enter it here for this session.",
    )
    st.session_state["openai_api_base"] = st.text_input(
        "OpenAI API Base URL (optional)",
        value=st.session_state.get("openai_api_base") or "",
        help="Leave blank to use the default OpenAI endpoint, or enter your "
             "enterprise-approved LLM gateway URL.",
    )

    st.divider()
    st.caption(f"Database file: `{DB_PATH}`")
    if os.path.exists(DB_PATH):
        st.success("Database found ✔")
    else:
        st.error("Database file not found. Place `credit_risk_portfolio.db` "
                 "in the app's working directory.")

    st.divider()
    st.caption(
        "This tool runs in **read-only** mode. Generated SQL is restricted to "
        "SELECT / WITH statements, validated before execution, and every "
        "answer is shown with its SQL, raw data, and a confidence score for "
        "auditability."
    )

if st.session_state["openai_api_key"]:
    os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
if st.session_state["openai_api_base"]:
    os.environ["OPENAI_BASE_URL"] = st.session_state["openai_api_base"]


# --------------------------------------------------------------------------
# Cached resources: LLMs and database connection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llms(api_key: str, api_base: str):
    """Set up the generation LLM and the (stricter) evaluator LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm, evaluator_llm


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    """Open a read-only connection to the analytical SQLite database."""
    # Read-only URI connection: the PoC must not perform writes.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return conn


# --------------------------------------------------------------------------
# Database schema (single source of truth passed to the LLMs)
# --------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# --------------------------------------------------------------------------
# Verified query library (pre-approved, tested, version-controlled templates)
# --------------------------------------------------------------------------
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql':  """SELECT
    sm.sector_name,
    SUM(lm.total_outstanding) / 1000000.0 AS total_outstanding_million,
    SUM(CASE
        WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
        THEN lm.total_outstanding
        ELSE 0
    END) / 1000000.0 AS npa_outstanding_million
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY
    sm.sector_name
ORDER BY
    total_outstanding_million DESC
"""
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': """SELECT
    loan_category,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_million,
    COUNT(loan_account_number) AS loan_count
FROM
    loan_master
GROUP BY
    loan_category
ORDER BY
    total_outstanding_million DESC
"""
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': """SELECT
    ifrs9_stage,
    COUNT(DISTINCT loan_account_number) AS loan_count,
    SUM(ead_amount) / 1000000.0 AS total_ead_million,
    SUM(ecl_amount) / 1000000.0 AS total_ecl_million
FROM
    provisioning
WHERE
    reporting_date = '2025-09-30'
GROUP BY
    ifrs9_stage
ORDER BY
    ifrs9_stage
"""
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': """SELECT
    sm.sector_name,
    AVG(p.provision_coverage_ratio) AS average_provision_coverage_ratio
FROM
    provisioning p
JOIN
    loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
WHERE
    p.reporting_date = '2025-09-30'
GROUP BY
    sm.sector_name
ORDER BY
    average_provision_coverage_ratio DESC
"""
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': """SELECT
    lm.borrower_name,
    sm.sector_name,
    lm.total_outstanding / 1000000.0 AS total_outstanding_million,
    lm.asset_classification
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY
    total_outstanding_million DESC
LIMIT 10
"""
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': """SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_million
FROM
    loan_master
WHERE
    group_name IS NOT NULL
GROUP BY
    group_name
ORDER BY
    total_outstanding_million DESC
LIMIT 5
"""
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': """SELECT
    loan_account_number,
    borrower_name,
    sm.sector_name,
    total_outstanding / 1000000.0 AS total_outstanding_million,
    days_past_due,
    asset_classification
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
WHERE
    days_past_due > 0
ORDER BY
    days_past_due DESC
"""
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_million
FROM
    loan_master
GROUP BY
    dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 0
        WHEN '1-30' THEN 1
        WHEN '31-60' THEN 2
        WHEN '61-90' THEN 3
        ELSE 4
    END
"""
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM
    borrower_rating
WHERE
    rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY
    pd_estimate DESC
"""
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': """SELECT
    reporting_date,
    SUM(ecl_amount) / 1000000.0 AS total_ecl_million
FROM
    provisioning
GROUP BY
    reporting_date
ORDER BY
    reporting_date
"""
    }
}


# --------------------------------------------------------------------------
# Pipeline stages (ported from the notebook, logic unchanged)
# --------------------------------------------------------------------------
def classify_intent(user_question, query_library, llm):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.
    - llm: The LangChain chat model used for classification.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### Role
You are a query router for a credit risk analytics team. Your job is to decide whether the business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### Input
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### Instructions
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about the shape of the answer: a question asking for row-level detail should not be matched to an aggregate template, and vice versa.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.
    - llm: The LangChain chat model used for generation.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### Role
You are a senior SQL developer specializing in credit risk analytics on a SQLite database.

### Input
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### Instructions
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, REPLACE, or ATTACH.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Ensure the query is SQLite compatible.
5. In SQLite, never subtract DATE() or date columns directly.
6. Alias every numeric column with a suffix that states its unit, so the result is self-describing.

### OUTPUT
Return ONLY a valid SQL query. Do not include any other text.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - evaluator_llm: The LangChain chat model used for the relevance check.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
### Role
You are a senior data validator. Your job is to check whether a SQL query correctly
answers a business user's question about credit risk.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### Instructions
Assess whether the SQL genuinely answers what the user asked, considering:
1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.
    - llm: The LangChain chat model used for the retry.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### Role
You are a senior SQL developer fixing a query that failed validation.

### Input
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - llm: The LangChain chat model used for narrative generation.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### Role
You are a senior credit risk analyst summarizing query results for a business user.

### Input
User Question:
{user_question}

Query Result Data:
{dataframe.to_string()}

### Instructions
1. Answer the user's question directly, using only the data shown above.
2. Be concise and focused on what the user actually asked, even if the data contains
   more rows/columns than strictly needed (e.g., a verified template may return the
   full portfolio breakdown).
3. Reference specific figures from the data where helpful.
4. Do not fabricate any figures that are not present in the data.

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - llm: Generation LLM.
    - evaluator_llm: Evaluator LLM used for the relevance check.
    - verbose (bool): If True, appends intermediate pipeline stages to the trace log.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    trace = []  # human-readable trace, mirrors the notebook's printed stages

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        trace.append(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        trace.append(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log['candidate_sql'] = candidate_sql

    if verbose:
        trace.append(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        trace.append(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            trace.append(f"    Failed check: {gate.get('failed_check')}")
            trace.append(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            trace.append(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context, llm)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log['gate_result'] = gate

        if verbose:
            trace.append(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                trace.append(f"    Retry failed check: {gate.get('failed_check')}")
                trace.append(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            trace.append(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        trace.append(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            trace.append(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], llm, log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        trace.append(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'trace': trace, **log, 'warnings': exec_result['warnings']}


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "audit_trail" not in st.session_state:
    st.session_state["audit_trail"] = []  # list of past run logs, newest last


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
st.caption(
    "Ask a routine commercial-lending portfolio question in plain English. "
    "Recurring questions are answered from pre-approved, version-controlled query "
    "templates; novel questions are answered with freshly generated, validated, "
    "read-only SQL. Every answer shows its SQL, raw data, and confidence score."
)

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["💬 Ask a Question", "📚 Verified Query Library", "✅ Batch Evaluation", "🧾 Audit Trail"]
)

# --------------------------------------------------------------------------
# Tab 1: Ask a question
# --------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a portfolio question")

    sample_questions = [
        "— Type your own question below —",
        "What is the sector-wise exposure and NPA breakdown across the portfolio?",
        "Show me the total portfolio outstanding by loan category.",
        "Give me the IFRS 9 stage-wise summary for the latest quarter.",
        "What is the average provision coverage ratio by sector?",
        "Who are the top 10 largest borrowers by outstanding amount?",
        "Which business groups have the largest exposure?",
        "List all overdue loan accounts.",
        "What is the days-past-due aging profile of the portfolio?",
        "Which borrowers were downgraded in the latest rating cycle?",
        "How has expected credit loss trended over the last four quarters?",
    ]
    picked = st.selectbox("Sample questions (optional)", sample_questions)

    default_text = "" if picked == sample_questions[0] else picked
    user_question = st.text_area(
        "Your question",
        value=default_text,
        height=90,
        placeholder="e.g. What is the total exposure in the Real Estate sector that is currently overdue?",
    )

    run_clicked = st.button("Run Query", type="primary", use_container_width=False)

    if run_clicked:
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state["openai_api_key"]:
            st.error("An OpenAI API key is required. Add one in the sidebar.")
        elif not os.path.exists(DB_PATH):
            st.error(f"Database file `{DB_PATH}` was not found in the app directory.")
        else:
            try:
                llm, evaluator_llm = get_llms(
                    st.session_state["openai_api_key"], st.session_state["openai_api_base"]
                )
                conn = get_connection(DB_PATH)

                with st.spinner("Running the query engine pipeline..."):
                    result = run_pipeline(
                        user_question, conn, verified_query_library, database_schema,
                        llm, evaluator_llm, verbose=True
                    )

                st.session_state["audit_trail"].append(result["log"])

                # ---- Route / classification summary ----
                col1, col2, col3 = st.columns(3)
                col1.metric("Route", result["route"])
                col2.metric("Query ID", result["query_id"] or "—")
                conf = result["confidence"]
                col3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))

                if result["match_reason"]:
                    st.caption(f"Routing rationale: {result['match_reason']}")

                if result["log"]["escalated"]:
                    st.error(result["narrative"])
                else:
                    # ---- Narrative answer ----
                    st.markdown("#### Answer")
                    st.write(result["narrative"])

                    if result.get("warnings"):
                        st.warning("Data quality warnings: " + "; ".join(result["warnings"]))

                    # ---- SQL used ----
                    st.markdown("#### SQL Used")
                    st.code(result["executed_sql"], language="sql")
                    if result["log"]["retry_used"]:
                        st.caption("Note: the initial SQL failed validation and was regenerated once before execution.")

                    # ---- Raw data ----
                    st.markdown("#### Raw Result Data")
                    st.dataframe(result["dataframe"], use_container_width=True)
                    st.caption(f"{result['row_count']} row(s) returned.")

                    csv_bytes = result["dataframe"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download result as CSV",
                        data=csv_bytes,
                        file_name="query_result.csv",
                        mime="text/csv",
                    )

                # ---- Full pipeline trace (auditability) ----
                with st.expander("Show full pipeline trace / audit log"):
                    for line in result["trace"]:
                        st.text(line)
                    st.json(result["log"])

            except Exception as e:
                st.error(f"Pipeline error: {e}")

# --------------------------------------------------------------------------
# Tab 2: Verified query library
# --------------------------------------------------------------------------
with tab_library:
    st.subheader("Pre-approved, version-controlled query templates")
    st.caption(
        f"{len(verified_query_library)} verified templates are available. These are tested, "
        "reviewed queries that recurring questions are routed to automatically."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"], language="sql")
            if st.button(f"Preview results for {qid}", key=f"preview_{qid}"):
                if os.path.exists(DB_PATH):
                    conn = get_connection(DB_PATH)
                    try:
                        preview_df = pd.read_sql_query(entry["sql"], conn)
                        st.dataframe(preview_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not run this template: {e}")
                else:
                    st.error(f"Database file `{DB_PATH}` was not found.")

# --------------------------------------------------------------------------
# Tab 3: Batch evaluation against ground-truth test cases
# --------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluate the pipeline against ground-truth test cases")
    st.caption(
        "Upload the `test_queries.csv` evaluation set (or use the copy bundled with the app, "
        "if present) to check routing accuracy, verified-query-match accuracy, and average "
        "confidence, mirroring the notebook's evaluation cells."
    )

    uploaded_csv = st.file_uploader("Upload test_queries.csv", type=["csv"])
    ground_truth_df = None
    if uploaded_csv is not None:
        ground_truth_df = pd.read_csv(uploaded_csv)
    elif os.path.exists(TEST_QUERIES_PATH):
        ground_truth_df = pd.read_csv(TEST_QUERIES_PATH)

    if ground_truth_df is not None:
        st.dataframe(ground_truth_df, use_container_width=True)

        if st.button("Run batch evaluation", type="primary"):
            if not st.session_state["openai_api_key"]:
                st.error("An OpenAI API key is required. Add one in the sidebar.")
            elif not os.path.exists(DB_PATH):
                st.error(f"Database file `{DB_PATH}` was not found in the app directory.")
            else:
                llm, evaluator_llm = get_llms(
                    st.session_state["openai_api_key"], st.session_state["openai_api_base"]
                )
                conn = get_connection(DB_PATH)

                evaluation_rows = []
                progress = st.progress(0.0)
                n = len(ground_truth_df)

                for i, (_, gt) in enumerate(ground_truth_df.iterrows()):
                    tr = run_pipeline(
                        gt["User Query"], conn, verified_query_library, database_schema,
                        llm, evaluator_llm, verbose=False
                    )
                    st.session_state["audit_trail"].append(tr["log"])

                    expected_query_id = gt.get("Expected Query ID")
                    query_id_match = (
                        (pd.isna(expected_query_id) and tr["query_id"] is None)
                        or tr["query_id"] == expected_query_id
                    )

                    evaluation_rows.append({
                        "Test Case": gt.get("Test Case"),
                        "User Query": gt.get("User Query"),
                        "Expected Route": gt.get("Expected Route"),
                        "Actual Route": tr["route"],
                        "Route Match": tr["route"] == gt.get("Expected Route"),
                        "Expected Query ID": expected_query_id,
                        "Actual Query ID": tr["query_id"],
                        "Query ID Match": query_id_match,
                        "Confidence": tr["confidence"],
                        "Rows Returned": tr["row_count"],
                    })
                    progress.progress((i + 1) / n)

                evaluation_df = pd.DataFrame(evaluation_rows)
                st.markdown("#### Evaluation Results")
                st.dataframe(evaluation_df, use_container_width=True)

                path_accuracy = evaluation_df["Route Match"].mean() * 100

                verified_mask = evaluation_df["Expected Route"].astype(str).str.strip().str.lower() == "verified"
                if verified_mask.any():
                    query_accuracy = evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100
                else:
                    query_accuracy = float("nan")

                numeric_confidence = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
                average_confidence = numeric_confidence.mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%" if pd.notna(query_accuracy) else "n/a")
                m3.metric("Average Confidence Score", f"{average_confidence:.2f}" if pd.notna(average_confidence) else "n/a")
    else:
        st.info("No test_queries.csv found. Upload one to run a batch evaluation.")

# --------------------------------------------------------------------------
# Tab 4: Session audit trail
# --------------------------------------------------------------------------
with tab_audit:
    st.subheader("Session audit trail")
    st.caption("Every question answered in this session, for review and auditability.")
    if st.session_state["audit_trail"]:
        audit_df = pd.DataFrame(st.session_state["audit_trail"])
        st.dataframe(audit_df, use_container_width=True)
        st.download_button(
            "Download audit trail as CSV",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
        if st.button("Clear audit trail"):
            st.session_state["audit_trail"] = []
            st.rerun()
    else:
        st.info("No queries have been run yet in this session.")
