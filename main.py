from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_ollama import OllamaLLM
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "menu"
MENU_COLLECTION = "menuItems"
CHAT_COLLECTION = "chat_history"
USER_COLLECTION = "users"  

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
menu_collection = db[MENU_COLLECTION]
chat_collection = db[CHAT_COLLECTION]
user_collection = db[USER_COLLECTION]

llm = OllamaLLM(model="gemma2:2b")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    email: EmailStr
    password: str

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

@app.post("/signup")
async def signup(user: User):
    existing_user = await user_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)
    await user_collection.insert_one({"email": user.email, "password": hashed_password})

    return {"message": "User registered successfully"}

@app.post("/login")
async def login(user: User):
    db_user = await user_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {"message": "Login successful"}

async def get_chat_history(user_id: str, limit: int = 5) -> List[str]:
    """Fetch the last few messages of a user to maintain context."""
    history = await chat_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit).to_list(length=limit)
    return [msg["message"] for msg in history][::-1]

@app.get("/chat")
async def chat(user_id: str, query: str = Query(...)):
    """Chatbot API with strict menu enforcement and session consistency."""
    menu_data = await menu_collection.find_one({}, {"_id": 0})  
    if not menu_data:
        return {"response": "Menu database not found."}

    chat_history = await get_chat_history(user_id)
    chat_context = "\n".join(chat_history)

    prompt = f"""
    You are a restaurant ordering assistant. Your goal is to *strictly follow the provided menu* and maintain consistency in conversation.
    
    Menu Data: {menu_data}
    
    Rules:
    1. If an item is not in the menu, do *not* later say it's available.
    2. If the user follows up on a past answer, ensure your response is logically consistent.
    3. Never contradict your previous statements.
    4. Avoid unnecessary greetings after the first message.
    5. Keep responses short and professional.

    Chat History:
    {chat_context}

    User Query: {query}
    """
    ai_response = await llm.ainvoke(prompt)
    await chat_collection.insert_one({
        "user_id": user_id,
        "message": query,
        "response": ai_response,
        "timestamp": datetime.utcnow()
    })

    return {"response": ai_response}
