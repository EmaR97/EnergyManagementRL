import logging
from functools import wraps

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (CallbackContext, MessageHandler, Application, filters, CallbackQueryHandler, CommandHandler)

from energymanagementrl.fusion_solar_connector import FusionSolarExceptionExtended
from energymanagementrl.rl import EnergyManagementSystem


class TelegramBot:
    def __init__(self, system: EnergyManagementSystem, token: str, allowed_users: list, logger=None):
        self.system = system
        self.token = token
        self.allowed_users = set(allowed_users)  # Use a set for O(1) lookups
        self.logger = logger or logging.getLogger(__name__)
        self.app = Application.builder().token(token).build()
        self._setup_handlers()

        self.logger.warning("TelegramBot initialized.")

    def _setup_handlers(self):
        handlers = [CommandHandler("start", self.handle_help), CommandHandler("help", self.handle_help),
                    CommandHandler("get_id", self.handle_get_id),
                    CommandHandler("set_controller_status", self.handle_set_controller_status),
                    CommandHandler("get_controller_status", self.handle_get_controller_status),
                    CommandHandler("get_battery_mode", self.handle_get_battery_mode),
                    CommandHandler("execute_control", self.handle_execute_control),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_invalid),
                    CallbackQueryHandler(self.button_callback), ]
        for handler in handlers:
            self.app.add_handler(handler)

        self.logger.warning("Handlers have been set up.")

    @staticmethod
    def authorized_only(func):
        @wraps(func)
        async def wrapper(self, update: Update, context: CallbackContext, *args, **kwargs):
            user_id = update.message.from_user.id
            self.logger.warning(f"Authorization check for user {user_id}.")
            if not self._is_authorized(user_id):
                self.logger.warning(f"Unauthorized access attempt by user {user_id}.")
                await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
                return
            return await func(self, update, context, *args, **kwargs)

        return wrapper

    def _is_authorized(self, user_id: int) -> bool:
        """Check if the user is authorized."""
        return user_id in self.allowed_users

    @staticmethod
    async def handle_help(update: Update, context: CallbackContext):
        help_text = ("Available Commands:\n"
                     "/help - Show this message\n"
                     "/get_controller_status - Get the current controller status\n"
                     "/set_controller_status - Set the controller's active status\n"
                     "/get_battery_mode - Get the current battery mode\n"
                     "/execute_control - Execute control iteration\n")
        await update.message.reply_text(help_text)

    @staticmethod
    async def handle_get_id(update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        await update.message.reply_text(f"{user_id}")
        context.bot.logger.warning(f"User {user_id} requested their ID.")

    @authorized_only
    async def handle_set_controller_status(self, update: Update, context: CallbackContext):
        self.logger.warning(f"User {update.message.from_user.id} is setting controller status.")
        keyboard = [[InlineKeyboardButton("Active", callback_data="set_controller_active"),
                     InlineKeyboardButton("Passive", callback_data="set_controller_passive")]]
        await update.message.reply_text("Choose the controller status:", reply_markup=InlineKeyboardMarkup(keyboard))

    @authorized_only
    async def handle_get_controller_status(self, update: Update, context: CallbackContext):
        status = "Active" if self.system.get_active() else "Passive"
        self.logger.warning(f"User {update.message.from_user.id} requested controller status: {status}.")
        await update.message.reply_text(f"Controller Status: {status}")

    @authorized_only
    async def handle_get_battery_mode(self, update: Update, context: CallbackContext):
        mode = self.system.get_last_battery_mode()
        self.logger.warning(f"User {update.message.from_user.id} requested battery mode: {mode}.")
        await update.message.reply_text(f"Current Battery Mode: {mode}")

    @authorized_only
    async def handle_execute_control(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        self.logger.warning(f"User {user_id} initiated control execution.")
        await update.message.reply_text("Execution started...")
        try:
            await self._execute_control_async()
            result = self.system.get_last_battery_mode()
            self.logger.warning(f"Execution control completed successfully. Result: {result}.")
            await update.message.reply_text(f"Execution control result: {result}")
        except FusionSolarExceptionExtended as e:
            self.logger.error(f"Execution control failed for user {user_id}: {e.code}")
            await update.message.reply_text(f"Execution control failed: {e.code}")

    async def _execute_control_async(self):
        """Execute control asynchronously to avoid blocking."""
        await self.system.execute_control()

    @authorized_only
    async def handle_invalid(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        self.logger.warning(f"User {user_id} entered an invalid command.")
        await update.message.reply_text("Invalid command! Use '/help' for a list of commands.")

    async def button_callback(self, update: Update, context: CallbackContext):
        query = update.callback_query
        user_id = query.from_user.id

        self.logger.warning(f"User {user_id} triggered button callback: {query.data}.")

        if not self._is_authorized(user_id):
            self.logger.warning(f"Unauthorized user {user_id} attempted to use a button callback.")
            await query.answer("Unauthorized access!", show_alert=True)
            return

        await query.answer()

        if query.data == "set_controller_active":
            self.system.set_active(True)
            self.logger.warning(f"User {user_id} set controller status to Active.")
            await query.edit_message_text("Controller status set to Active!")
        elif query.data == "set_controller_passive":
            self.system.set_active(False)
            self.logger.warning(f"User {user_id} set controller status to Passive.")
            await query.edit_message_text("Controller status set to Passive!")

    def run(self):
        self.logger.warning("Bot is starting...")
        self.app.run_polling()
        self.logger.warning("Bot has stopped.")
