import asyncio
import logging
import os
import sys
import time

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN:          str = os.getenv("BOT_TOKEN", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
MINI_APP_URL:       str = os.getenv("MINI_APP_URL", "https://curious-pudd.netlify.app/")

if not BOT_TOKEN:
    raise EnvironmentError("BOT_TOKEN is not set in .env")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")

# ── Constants ──────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "gpt":      "openai/gpt-4o-mini",
    "claude":   "anthropic/claude-3-haiku",
    "deepseek": "deepseek/deepseek-chat",
    "llama":    "meta-llama/llama-3-70b-instruct",
}

MODEL_NAMES: dict[str, str] = {
    "gpt":      "GPT-4o mini",
    "claude":   "Claude Haiku",
    "deepseek": "DeepSeek Chat",
    "llama":    "Llama 3 70B",
}

SYSTEM_PROMPT = (
    "Ты умный и дружелюбный AI-ассистент в Telegram. "
    "Учитывай весь контекст диалога. Отвечай полезно, чётко и по делу. "
    "Для форматирования кода используй обратные кавычки."
)

MAX_HISTORY        = 20    # сообщений в памяти (включая system)
RATE_LIMIT_SECONDS = 3     # минимум секунд между запросами
MAX_TG_LENGTH      = 4000  # лимит Telegram (с запасом от 4096)
STREAM_UPDATE_CHARS = 80   # обновлять сообщение каждые N символов

# ── Clients ────────────────────────────────────────────────────────────────
ai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)

dp = Dispatcher()

# ── In-memory storage (замените на БД для продакшна) ──────────────────────
user_models:   dict[int, str]        = {}
user_memory:   dict[int, list[dict]] = {}
user_personas: dict[int, str]        = {}
user_last_req: dict[int, float]      = {}

# ── Helpers ────────────────────────────────────────────────────────────────

def get_model_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 GPT-4o mini",   callback_data="gpt")],
        [InlineKeyboardButton(text="🧠 Claude Haiku",  callback_data="claude")],
        [InlineKeyboardButton(text="🐋 DeepSeek Chat", callback_data="deepseek")],
        [InlineKeyboardButton(text="🦙 Llama 3 70B",   callback_data="llama")],
    ])


def init_memory(user_id: int) -> None:
    """Инициализирует историю с персональным или дефолтным промптом."""
    persona = user_personas.get(user_id, SYSTEM_PROMPT)
    user_memory[user_id] = [{"role": "system", "content": persona}]


def trim_memory(user_id: int) -> None:
    """Обрезает историю, сохраняя system + последние N сообщений."""
    h = user_memory.get(user_id, [])
    if len(h) > MAX_HISTORY:
        user_memory[user_id] = [h[0]] + h[-(MAX_HISTORY - 1):]


def rate_limited(user_id: int) -> float:
    """Возвращает оставшееся время ожидания (0 = можно отправлять)."""
    return max(0.0, RATE_LIMIT_SECONDS - (time.time() - user_last_req.get(user_id, 0)))


def split_text(text: str) -> list[str]:
    """Разбивает длинный текст на части для Telegram."""
    if len(text) <= MAX_TG_LENGTH:
        return [text]
    parts = []
    while text:
        if len(text) <= MAX_TG_LENGTH:
            parts.append(text)
            break
        idx = text.rfind("\n", 0, MAX_TG_LENGTH)
        if idx == -1:
            idx = MAX_TG_LENGTH
        parts.append(text[:idx])
        text = text[idx:].lstrip()
    return parts


async def stream_response(model: str, messages: list[dict], placeholder: Message) -> str:
    """
    Асинхронный стриминг от OpenRouter.
    Обновляет placeholder-сообщение каждые STREAM_UPDATE_CHARS символов.
    """
    full_text = ""
    last_edit_len = 0

    stream = await ai_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=3000,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta

        if len(full_text) - last_edit_len >= STREAM_UPDATE_CHARS:
            try:
                await placeholder.edit_text(full_text + " ✍️")
                last_edit_len = len(full_text)
            except Exception:
                pass  # слишком частые апдейты иногда вызывают FloodWait

    return full_text


def format_error(err: str) -> str:
    """Возвращает понятное сообщение об ошибке."""
    if "401" in err:
        return "❌ Проблема с API-ключом. Обратитесь к администратору."
    if "429" in err:
        return "⏳ Слишком много запросов. Подождите минуту и попробуйте снова."
    if "400" in err and "valid model" in err.lower():
        return "❌ Эта модель временно недоступна. Попробуй другую — /model"
    if "503" in err or "timeout" in err.lower():
        return "🔌 Сервис AI временно недоступен. Попробуйте позже."
    return f"⚠️ Ошибка: `{err[:200]}`"

# ── Handlers ───────────────────────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "👋 *Добро пожаловать в AI Assistant!*\n\n"
        "Общайтесь с лучшими AI-моделями прямо в Telegram.\n"
        "Выберите модель для начала:",
        reply_markup=get_model_keyboard(),
    )


@dp.message(Command("model"))
async def cmd_model(msg: Message) -> None:
    await msg.answer("🔄 Выберите модель:", reply_markup=get_model_keyboard())


@dp.message(Command("app"))
async def cmd_app(msg: Message) -> None:
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🚀 Открыть Mini App",
            web_app=WebAppInfo(url=MINI_APP_URL),
        )
    ]])
    await msg.answer(
        "Откройте полноценный интерфейс с историей диалога:",
        reply_markup=kb,
    )


@dp.message(Command("reset"))
async def cmd_reset(msg: Message) -> None:
    init_memory(msg.from_user.id)
    await msg.answer("🗑 *История диалога очищена.* Начнём заново!")


@dp.message(Command("persona"))
async def cmd_persona(msg: Message) -> None:
    uid = msg.from_user.id
    parts = msg.text.split(maxsplit=1)

    if len(parts) < 2:
        current = user_personas.get(uid, SYSTEM_PROMPT)
        await msg.answer(
            f"🎭 *Текущая роль бота:*\n`{current}`\n\n"
            "Чтобы изменить:\n`/persona Ты опытный Python-разработчик...`\n\n"
            "Сбросить до стандартной: `/persona reset`"
        )
        return

    new_persona = parts[1].strip()

    if new_persona.lower() == "reset":
        user_personas.pop(uid, None)
        init_memory(uid)
        await msg.answer("♻️ Роль сброшена до стандартной. История очищена.")
        return

    user_personas[uid] = new_persona
    init_memory(uid)
    await msg.answer(
        f"✅ *Роль обновлена:*\n`{new_persona}`\n\n"
        "_История диалога сброшена._"
    )


@dp.message(Command("history"))
async def cmd_history(msg: Message) -> None:
    uid = msg.from_user.id
    history = user_memory.get(uid, [])
    dialog = [m for m in history if m["role"] != "system"]

    if not dialog:
        await msg.answer("💬 *История диалога пуста.*")
        return

    lines = []
    for m in dialog[-10:]:
        icon = "👤" if m["role"] == "user" else "🤖"
        preview = m["content"][:250].replace("`", "'")
        if len(m["content"]) > 250:
            preview += "…"
        lines.append(f"{icon} *{m['role']}:*\n{preview}")

    await msg.answer("📜 *Последние сообщения:*\n\n" + "\n\n".join(lines))


@dp.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "📖 *Команды бота:*\n\n"
        "/start — выбор AI-модели\n"
        "/model — сменить модель\n"
        "/reset — очистить историю диалога\n"
        "/persona — задать роль боту\n"
        "/history — последние сообщения\n"
        "/app — открыть Mini App\n"
        "/help — эта справка\n\n"
        "_Просто пишите сообщение — бот ответит!_"
    )


@dp.callback_query(F.data.in_(MODELS.keys()))
async def cb_model(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    key = cb.data

    user_models[uid] = MODELS[key]
    init_memory(uid)

    await cb.message.edit_text(
        f"✅ *Выбрана модель: {MODEL_NAMES[key]}*\n\n"
        "Теперь просто пишите сообщение!\n\n"
        "Полезные команды:\n"
        "/model — сменить модель\n"
        "/reset — очистить историю\n"
        "/persona — задать роль боту\n"
        "/app — открыть Mini App"
    )
    await cb.answer(f"✓ {MODEL_NAMES[key]}")


@dp.message(F.text)
async def on_message(msg: Message) -> None:
    uid = msg.from_user.id

    # Проверяем выбор модели
    if uid not in user_models:
        await msg.answer("⚠️ Сначала выберите модель — /start")
        return

    # Rate limiting
    wait = rate_limited(uid)
    if wait > 0:
        await msg.answer(f"⏳ Подождите {wait:.1f} сек. перед следующим сообщением.")
        return

    user_last_req[uid] = time.time()

    # Инициализируем память если нужно
    if uid not in user_memory:
        init_memory(uid)

    # Добавляем сообщение пользователя
    user_memory[uid].append({"role": "user", "content": msg.text})

    # Показываем индикатор
    placeholder = await msg.answer("🤔 *Думаю...*")

    try:
        model  = user_models[uid]
        logger.info("User %d → model %s", uid, model)

        answer = await stream_response(model, user_memory[uid], placeholder)

        if not answer.strip():
            raise ValueError("Модель вернула пустой ответ")

        # Сохраняем ответ и обрезаем историю
        user_memory[uid].append({"role": "assistant", "content": answer})
        trim_memory(uid)

        # Отправляем (разбиваем если длинный)
        parts = split_text(answer)
        await placeholder.edit_text(parts[0])

        for part in parts[1:]:
            await asyncio.sleep(0.3)
            await msg.answer(part)

    except Exception as e:
        err_str = str(e)
        logger.error("User %d error: %s", uid, err_str)

        # Убираем неудачное сообщение из памяти
        if user_memory[uid] and user_memory[uid][-1]["role"] == "user":
            user_memory[uid].pop()

        await placeholder.edit_text(format_error(err_str))

# ── Entry point ────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("🤖 AI Assistant Bot is starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
