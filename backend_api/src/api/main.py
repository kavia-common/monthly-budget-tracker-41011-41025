from datetime import date as Date, datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlmodel import Field as SQLField, Relationship, Session, SQLModel, create_engine, select

# Application metadata and initialization with CORS.
app = FastAPI(
    title="Monthly Budget Tracker API",
    description="REST API for authentication, transactions, budgets, categories, and dashboard summaries.",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# For demo/preview: allow all origins. In production, restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PySecure-4-Minimal: permissive only for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup (SQLite file in container)
DATABASE_URL = "sqlite:///./budget.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# =========================
# Database Models (SQLModel)
# =========================
class User(SQLModel, table=True):
    """User model for local auth (demo token)."""
    id: Optional[int] = SQLField(default=None, primary_key=True)
    email: str = SQLField(index=True, unique=True)
    password_hash: str


class Category(SQLModel, table=True):
    """Category for transactions and budgets.

    Note:
    - Database column is 'category_type' to avoid clashing with the built-in/type annotation name 'type'.
    - API response schemas still expose field 'type' for backward compatibility.
    """
    id: Optional[int] = SQLField(default=None, primary_key=True)
    name: str = SQLField(index=True)
    category_type: str = SQLField(index=True)  # "income" | "expense"

    # Use forward-ref strings for relationships to avoid runtime annotation resolution issues
    transactions: List["Transaction"] = Relationship(back_populates="category")
    budgets: List["Budget"] = Relationship(back_populates="category")


class Transaction(SQLModel, table=True):
    """Financial transaction tied to a category and a month."""
    id: Optional[int] = SQLField(default=None, primary_key=True)
    date: Date
    amount: float
    note: Optional[str] = None
    month: str = SQLField(index=True)  # YYYY-MM

    category_id: int = SQLField(foreign_key="category.id")
    category: Optional[Category] = Relationship(back_populates="transactions")


class Budget(SQLModel, table=True):
    """Budget limit per category per month; spent is computed via transactions."""
    id: Optional[int] = SQLField(default=None, primary_key=True)
    month: str = SQLField(index=True)  # YYYY-MM
    category_id: int = SQLField(foreign_key="category.id")
    limit: float

    category: Optional[Category] = Relationship(back_populates="budgets")


def create_db_and_tables() -> None:
    """Create database tables."""
    SQLModel.metadata.create_all(engine)


# Dependency to get DB session
def get_session() -> Session:
    """Context-managed SQLModel session."""
    with Session(engine) as session:
        yield session


# =========================
# Pydantic Schemas
# =========================
class TokenResponse(BaseModel):
    """Simple token response for mock local auth."""
    token: str = Field(..., description="Mock token to be echoed by client for demo use.")


class LoginRequest(BaseModel):
    """Login request model for demo local auth."""
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or len(v) < 4:
            raise ValueError("Password must be at least 4 characters for demo.")
        return v


class CategoryRead(BaseModel):
    """Category response schema."""
    id: int
    name: str
    type: str


class TransactionBase(BaseModel):
    """Base transaction schema for create/update."""
    date: Date = Field(..., description="Transaction date in ISO format (YYYY-MM-DD).")
    amount: float = Field(..., description="Positive for income, negative for expense optional; we still rely on category type.")
    category_id: int = Field(..., description="Existing category ID.")
    note: Optional[str] = Field(None, max_length=200)
    month: str = Field(..., description="Month in YYYY-MM format.")

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError as exc:
            raise ValueError("month must be in YYYY-MM format") from exc
        return v


class TransactionCreate(TransactionBase):
    """Create transaction payload."""
    pass


class TransactionUpdate(BaseModel):
    """Update transaction payload (partial allowed)."""
    date: Optional[Date] = None
    amount: Optional[float] = None
    category_id: Optional[int] = None
    note: Optional[str] = Field(None, max_length=200)
    month: Optional[str] = None

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError as exc:
            raise ValueError("month must be in YYYY-MM format") from exc
        return v


class TransactionRead(BaseModel):
    """Transaction read schema."""
    id: int
    date: Date
    amount: float
    category_id: int
    note: Optional[str]
    month: str


class BudgetBase(BaseModel):
    """Base budget schema for create/update."""
    month: str = Field(..., description="Month in YYYY-MM format.")
    category_id: int = Field(..., description="Existing category ID.")
    limit: float = Field(..., ge=0, description="Budget limit for the category in the month.")

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError as exc:
            raise ValueError("month must be in YYYY-MM format") from exc
        return v


class BudgetCreate(BudgetBase):
    """Create budget payload."""
    pass


class BudgetUpdate(BaseModel):
    """Update budget payload (partial allowed)."""
    month: Optional[str] = None
    category_id: Optional[int] = None
    limit: Optional[float] = Field(None, ge=0)

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError as exc:
            raise ValueError("month must be in YYYY-MM format") from exc
        return v


class BudgetRead(BaseModel):
    """Budget read schema including computed spent amount."""
    id: int
    month: str
    category_id: int
    limit: float
    spent: float


class DashboardTimeseriesPoint(BaseModel):
    """Timeseries point for a day's net or separate income/expense sums."""
    day: int
    income: float
    expense: float


class DashboardSummary(BaseModel):
    """Dashboard summary for a given month."""
    month: str
    income_total: float
    expense_total: float
    balance: float
    top_categories: List[Dict[str, Any]]
    timeseries: List[DashboardTimeseriesPoint]


# =========================
# Utilities
# =========================
def _hash_demo_password(p: str) -> str:
    """Very simple demo hashing (NOT secure, for preview only)."""
    # PySecure-4-Minimal: No real hashing to avoid introducing deps; not for production.
    return f"demo-{len(p)}"


def current_month_str(dt: Optional[datetime] = None) -> str:
    """Return current month string YYYY-MM."""
    dt = dt or datetime.utcnow()
    return dt.strftime("%Y-%m")


def previous_month_str(dt: Optional[datetime] = None) -> str:
    """Return previous month string YYYY-MM."""
    dt = dt or datetime.utcnow()
    first = dt.replace(day=1)
    prev_last = first - timedelta(days=1)  # type: ignore[name-defined]
    return prev_last.strftime("%Y-%m")


# local timedelta import kept near usage to avoid namespace clutter
from datetime import timedelta  # noqa: E402  # isort:skip


def compute_spent_for_category_month(session: Session, category_id: int, month: str) -> float:
    """Compute total spent for a category in a given month (expenses as positive sum of absolute values)."""
    # We consider expense categories: sum of positive amounts for transactions categorized as expense.
    # For income categories, spent conceptually is 0 in budgets context.
    # Here we compute absolute of negative or positive based on category type.
    # Simpler: sum of amounts where category type is expense and month matches, only expenses (amount < 0 or any amount) -> use positive absolute.
    cat = session.get(Category, category_id)
    if not cat:
        return 0.0
    if cat.category_type != "expense":
        return 0.0

    stmt = select(Transaction).where(Transaction.category_id == category_id, Transaction.month == month)
    total = 0.0
    for tr in session.exec(stmt):
        # If amount is positive by mistake for expense, still count toward spent.
        total += abs(tr.amount)
    return round(total, 2)


def get_category_map(session: Session) -> Dict[int, Category]:
    """Return a map of category_id to Category."""
    stmt = select(Category)
    return {c.id: c for c in session.exec(stmt)}  # type: ignore[dict-item]


# =========================
# Routers
# =========================
api_router = APIRouter(prefix="/api", tags=["API"])


auth_router = APIRouter(prefix="/auth", tags=["auth"])


# PUBLIC_INTERFACE
@auth_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login (demo)",
    description="Mock local authentication that returns a placeholder token. No external services.",
)
def login(payload: LoginRequest, session: Session = Depends(get_session)) -> TokenResponse:
    """
    Demo login that checks for a user by email. If not present, auto-creates a user for the demo.
    Returns a simple token placeholder without external validation.
    """
    email = payload.email
    password = payload.password

    # Fetch or create user
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(email=email, password_hash=_hash_demo_password(password))
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        # Check password length only (demo); do not log sensitive data.
        if user.password_hash != _hash_demo_password(password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Simple placeholder token
    token = f"demo-token-{user.id}"
    return TokenResponse(token=token)


categories_router = APIRouter(prefix="/categories", tags=["categories"])


# PUBLIC_INTERFACE
@categories_router.get(
    "",
    response_model=List[CategoryRead],
    summary="List categories",
    description="Returns preset categories for income and expense.",
)
def list_categories(session: Session = Depends(get_session)) -> List[CategoryRead]:
    """Return all categories."""
    stmt = select(Category)
    items = session.exec(stmt).all()
    # Map underlying model 'category_type' to API field 'type'
    return [CategoryRead(id=c.id, name=c.name, type=c.category_type) for c in items]


transactions_router = APIRouter(prefix="/transactions", tags=["transactions"])


# PUBLIC_INTERFACE
@transactions_router.get(
    "",
    response_model=List[TransactionRead],
    summary="List transactions",
    description="Fetch transactions. Optionally filter by month (YYYY-MM).",
)
def list_transactions(
    month: Optional[str] = Query(None, description="Filter by YYYY-MM"),
    session: Session = Depends(get_session),
) -> List[TransactionRead]:
    """List transactions, optionally filtered by month."""
    stmt = select(Transaction)
    if month:
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM") from exc
        stmt = stmt.where(Transaction.month == month)
    items = session.exec(stmt).all()
    return [
        TransactionRead(
            id=t.id, date=t.date, amount=t.amount, category_id=t.category_id, note=t.note, month=t.month
        )
        for t in items
    ]


# PUBLIC_INTERFACE
@transactions_router.post(
    "",
    response_model=TransactionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create transaction",
)
def create_transaction(payload: TransactionCreate, session: Session = Depends(get_session)) -> TransactionRead:
    """Create a new transaction."""
    cat = session.get(Category, payload.category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    obj = Transaction(
        date=payload.date,
        amount=float(payload.amount),
        category_id=payload.category_id,
        note=payload.note,
        month=payload.month,
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return TransactionRead(
        id=obj.id, date=obj.date, amount=obj.amount, category_id=obj.category_id, note=obj.note, month=obj.month
    )


# PUBLIC_INTERFACE
@transactions_router.put(
    "/{transaction_id}",
    response_model=TransactionRead,
    summary="Update transaction",
)
def update_transaction(
    transaction_id: int, payload: TransactionUpdate, session: Session = Depends(get_session)
) -> TransactionRead:
    """Update an existing transaction."""
    obj = session.get(Transaction, transaction_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if payload.category_id is not None:
        cat = session.get(Category, payload.category_id)
        if not cat:
            raise HTTPException(status_code=404, detail="Category not found")
        obj.category_id = payload.category_id

    if payload.date is not None:
        obj.date = payload.date
    if payload.amount is not None:
        obj.amount = float(payload.amount)
    if payload.note is not None:
        obj.note = payload.note
    if payload.month is not None:
        obj.month = payload.month

    session.add(obj)
    session.commit()
    session.refresh(obj)
    return TransactionRead(
        id=obj.id, date=obj.date, amount=obj.amount, category_id=obj.category_id, note=obj.note, month=obj.month
    )


# PUBLIC_INTERFACE
@transactions_router.delete(
    "/{transaction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete transaction",
)
def delete_transaction(transaction_id: int, session: Session = Depends(get_session)) -> Response:
    """Delete a transaction by ID."""
    obj = session.get(Transaction, transaction_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Transaction not found")
    session.delete(obj)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


budgets_router = APIRouter(prefix="/budgets", tags=["budgets"])


# PUBLIC_INTERFACE
@budgets_router.get(
    "",
    response_model=List[BudgetRead],
    summary="List budgets",
    description="List budgets; optionally filter by month YYYY-MM.",
)
def list_budgets(
    month: Optional[str] = Query(None, description="Filter by YYYY-MM"),
    session: Session = Depends(get_session),
) -> List[BudgetRead]:
    """List budgets with computed spent values."""
    stmt = select(Budget)
    if month:
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM") from exc
        stmt = stmt.where(Budget.month == month)
    items = session.exec(stmt).all()

    out: List[BudgetRead] = []
    for b in items:
        spent = compute_spent_for_category_month(session, b.category_id, b.month)
        out.append(BudgetRead(id=b.id, month=b.month, category_id=b.category_id, limit=b.limit, spent=spent))
    return out


# PUBLIC_INTERFACE
@budgets_router.post(
    "",
    response_model=BudgetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create budget",
)
def create_budget(payload: BudgetCreate, session: Session = Depends(get_session)) -> BudgetRead:
    """Create a budget entry."""
    cat = session.get(Category, payload.category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    obj = Budget(month=payload.month, category_id=payload.category_id, limit=float(payload.limit))
    session.add(obj)
    session.commit()
    session.refresh(obj)
    spent = compute_spent_for_category_month(session, obj.category_id, obj.month)
    return BudgetRead(id=obj.id, month=obj.month, category_id=obj.category_id, limit=obj.limit, spent=spent)


# PUBLIC_INTERFACE
@budgets_router.put(
    "/{budget_id}",
    response_model=BudgetRead,
    summary="Update budget",
)
def update_budget(budget_id: int, payload: BudgetUpdate, session: Session = Depends(get_session)) -> BudgetRead:
    """Update a budget entry by ID."""
    obj = session.get(Budget, budget_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Budget not found")

    if payload.category_id is not None:
        cat = session.get(Category, payload.category_id)
        if not cat:
            raise HTTPException(status_code=404, detail="Category not found")
        obj.category_id = payload.category_id
    if payload.month is not None:
        obj.month = payload.month
    if payload.limit is not None:
        obj.limit = float(payload.limit)

    session.add(obj)
    session.commit()
    session.refresh(obj)
    spent = compute_spent_for_category_month(session, obj.category_id, obj.month)
    return BudgetRead(id=obj.id, month=obj.month, category_id=obj.category_id, limit=obj.limit, spent=spent)


# PUBLIC_INTERFACE
@budgets_router.delete(
    "/{budget_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete budget",
)
def delete_budget(budget_id: int, session: Session = Depends(get_session)) -> Response:
    """Delete a budget entry by ID."""
    obj = session.get(Budget, budget_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Budget not found")
    session.delete(obj)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _calc_income_expense_for_month(session: Session, month: str) -> Tuple[float, float]:
    """Return (income_total, expense_total) for the month."""
    cat_map = get_category_map(session)
    stmt = select(Transaction).where(Transaction.month == month)
    income_total = 0.0
    expense_total = 0.0
    for t in session.exec(stmt):
        cat = cat_map.get(t.category_id)
        if not cat:
            continue
        if cat.category_type == "income":
            income_total += abs(t.amount)
        else:
            expense_total += abs(t.amount)
    return round(income_total, 2), round(expense_total, 2)


def _calc_top_categories(session: Session, month: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Return top expense categories by spent amount."""
    cat_map = get_category_map(session)
    sums: Dict[int, float] = {}
    stmt = select(Transaction).where(Transaction.month == month)
    for t in session.exec(stmt):
        cat = cat_map.get(t.category_id)
        if not cat or cat.category_type != "expense":
            continue
        sums[t.category_id] = sums.get(t.category_id, 0.0) + abs(t.amount)
    # Sort desc
    top = sorted(sums.items(), key=lambda x: x[1], reverse=True)[:limit]
    result: List[Dict[str, Any]] = []
    for cid, total in top:
        cat = cat_map.get(cid)
        if cat:
            result.append({"category_id": cid, "category": cat.name, "spent": round(total, 2)})
    return result


def _calc_timeseries(session: Session, month: str) -> List[DashboardTimeseriesPoint]:
    """Compute day-wise income and expense totals for the month."""
    stmt = select(Transaction).where(Transaction.month == month)
    cat_map = get_category_map(session)
    by_day: Dict[int, Dict[str, float]] = {}
    for t in session.exec(stmt):
        day = t.date.day
        cat = cat_map.get(t.category_id)
        if not cat:
            continue
        if day not in by_day:
            by_day[day] = {"income": 0.0, "expense": 0.0}
        if cat.category_type == "income":
            by_day[day]["income"] += abs(t.amount)
        else:
            by_day[day]["expense"] += abs(t.amount)
    out: List[DashboardTimeseriesPoint] = []
    for d in sorted(by_day.keys()):
        out.append(DashboardTimeseriesPoint(day=d, income=round(by_day[d]["income"], 2), expense=round(by_day[d]["expense"], 2)))
    return out


# PUBLIC_INTERFACE
@dashboard_router.get(
    "/summary",
    response_model=DashboardSummary,
    summary="Dashboard summary for a month",
    description="Returns income total, expense total, balance, top categories, and a daily timeseries for the given month.",
)
def dashboard_summary(
    month: Optional[str] = Query(None, description="YYYY-MM, defaults to current month"),
    session: Session = Depends(get_session),
) -> DashboardSummary:
    """
    Aggregate dashboard for the specified month.

    Parameters:
    - month: YYYY-MM string; if omitted, uses current month.

    Returns:
    - DashboardSummary data model containing totals and timeseries.
    """
    m = month or current_month_str()
    try:
        datetime.strptime(m, "%Y-%m")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="month must be YYYY-MM") from exc

    income_total, expense_total = _calc_income_expense_for_month(session, m)
    balance = round(income_total - expense_total, 2)
    top = _calc_top_categories(session, m)
    series = _calc_timeseries(session, m)
    return DashboardSummary(
        month=m,
        income_total=income_total,
        expense_total=expense_total,
        balance=balance,
        top_categories=top,
        timeseries=series,
    )


seed_router = APIRouter(prefix="/seed", tags=["seed"])


def _ensure_seed_categories(session: Session) -> List[Category]:
    existing = session.exec(select(Category)).all()
    if existing:
        return existing
    preset = [
        Category(name="Salary", category_type="income"),
        Category(name="Freelance", category_type="income"),
        Category(name="Groceries", category_type="expense"),
        Category(name="Rent", category_type="expense"),
        Category(name="Utilities", category_type="expense"),
        Category(name="Transport", category_type="expense"),
        Category(name="Dining", category_type="expense"),
    ]
    for c in preset:
        session.add(c)
    session.commit()
    return session.exec(select(Category)).all()


def _seed_transactions_and_budgets(session: Session) -> None:
    cats = _ensure_seed_categories(session)
    cat_by_name = {c.name: c for c in cats}
    cm = current_month_str()
    pm = previous_month_str()

    # Avoid duplicate seed by checking presence
    if session.exec(select(Transaction).where(Transaction.month == cm)).first():
        return

    # Income
    income_tx = [
        Transaction(date=Date.fromisoformat(f"{cm}-01"), amount=5000.0, category_id=cat_by_name["Salary"].id, note="Monthly salary", month=cm),
        Transaction(date=Date.fromisoformat(f"{cm}-15"), amount=800.0, category_id=cat_by_name["Freelance"].id, note="Contract work", month=cm),
        Transaction(date=Date.fromisoformat(f"{pm}-01"), amount=5000.0, category_id=cat_by_name["Salary"].id, note="Last month salary", month=pm),
    ]

    # Expenses (negative or positive amounts; we take absolute in computations)
    expense_tx = [
        Transaction(date=Date.fromisoformat(f"{cm}-02"), amount=120.5, category_id=cat_by_name["Groceries"].id, note="Weekly groceries", month=cm),
        Transaction(date=Date.fromisoformat(f"{cm}-03"), amount=1500.0, category_id=cat_by_name["Rent"].id, note="Apartment rent", month=cm),
        Transaction(date=Date.fromisoformat(f"{cm}-05"), amount=85.75, category_id=cat_by_name["Utilities"].id, note="Electricity bill", month=cm),
        Transaction(date=Date.fromisoformat(f"{cm}-07"), amount=60.0, category_id=cat_by_name["Transport"].id, note="Monthly pass", month=cm),
        Transaction(date=Date.fromisoformat(f"{cm}-09"), amount=45.2, category_id=cat_by_name["Dining"].id, note="Dinner out", month=cm),
        Transaction(date=Date.fromisoformat(f"{pm}-04"), amount=130.0, category_id=cat_by_name["Groceries"].id, note="Last month groceries", month=pm),
        Transaction(date=Date.fromisoformat(f"{pm}-05"), amount=1500.0, category_id=cat_by_name["Rent"].id, note="Last month rent", month=pm),
    ]

    for t in income_tx + expense_tx:
        session.add(t)
    session.commit()

    # Seed budgets for common categories for current month
    budgets = [
        Budget(month=cm, category_id=cat_by_name["Groceries"].id, limit=500.0),
        Budget(month=cm, category_id=cat_by_name["Rent"].id, limit=1500.0),
        Budget(month=cm, category_id=cat_by_name["Utilities"].id, limit=200.0),
        Budget(month=cm, category_id=cat_by_name["Transport"].id, limit=120.0),
        Budget(month=cm, category_id=cat_by_name["Dining"].id, limit=150.0),
    ]
    for b in budgets:
        session.add(b)
    session.commit()


# PUBLIC_INTERFACE
@seed_router.post(
    "",
    summary="Seed demo data",
    description="Populate demo categories, transactions, and budgets for current and previous months.",
)
def seed(session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Seed the database with demo data. Idempotent for current month."""
    _ensure_seed_categories(session)
    _seed_transactions_and_budgets(session)
    return {"status": "ok", "message": "Seeded demo data"}


# Root health check
# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health check")
def health_check() -> Dict[str, str]:
    """Return health message."""
    return {"message": "Healthy"}


# Mount routers under /api
api_router.include_router(auth_router)
api_router.include_router(categories_router)
api_router.include_router(transactions_router)
api_router.include_router(budgets_router)
api_router.include_router(dashboard_router)
api_router.include_router(seed_router)
app.include_router(api_router)


# Startup event: create tables and auto-seed once.
@app.on_event("startup")
def on_startup() -> None:
    """Initialize database and ensure seed data is present."""
    create_db_and_tables()
    with Session(engine) as session:
        _ensure_seed_categories(session)
        _seed_transactions_and_budgets(session)
