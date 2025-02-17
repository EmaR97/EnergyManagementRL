import logging
from functools import wraps

from telegram import BotCommand, CallbackQuery
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import Conflict
from telegram.ext import (CallbackContext, MessageHandler, Application, filters, CallbackQueryHandler, CommandHandler)

from energymanagementrl.fusion_solar_connector import FusionSolarExceptionExtended, BatteryWorkingMode
from energymanagementrl.rl import EnergyManagementSystem


def get_logger(name: str):
    info_logger = logging.getLogger(f"{__name__}" + name)  # Unique logger name
    info_logger.setLevel(logging.INFO)  # Capture INFO and above

    # Ensure handler is added only once
    if not info_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)  # Ensure it logs INFO and above

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        info_logger.addHandler(handler)

    return info_logger


class TelegramBot:
    def __init__(self, system: EnergyManagementSystem, token: str, allowed_users: list, logger=None):
        self.system = system
        self.token = token
        self.allowed_users = set(allowed_users)  # Use a set for O(1) lookups
        self.logger = logger or get_logger("_info")
        self.app = Application.builder().token(token).build()
        self._setup_handlers()
        self.logger.info("TelegramBot initialized.")

    async def error_handler(self, update: object, context: CallbackContext) -> None:
        """Handles errors raised by the application."""
        if isinstance(context.error, Conflict):
            self.logger.warning("Bot instance conflict detected. Ignoring...")
        else:
            self.logger.exception(f"Unhandled error: {context.error}")

    @staticmethod
    def authorized_only(func):
        @wraps(func)
        async def wrapper(self, update: Update, context: CallbackContext, *args, **kwargs):
            user_id = update.message.from_user.id
            self.logger.info(f"Authorization check for user {user_id}.")
            if not self._is_authorized(user_id):
                self.logger.info(f"Unauthorized access attempt by user {user_id}.")
                await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
                return
            return await func(self, update, context, *args, **kwargs)

        return wrapper

    def _is_authorized(self, user_id: int) -> bool:
        """Check if the user is authorized."""
        return user_id in self.allowed_users

    COMMANDS = [
        ("help", "Show available commands"),
        ("get_controller_status", "Get the current controller status"),
        ("set_controller_status", "Set the controller's active status"),
        ("get_battery_mode", "Get the current battery mode"),
        ("get_battery_mode_direct", "Get the real battery mode"),
        ("set_battery_mode_direct", "Manually set the battery mode"),
        ("execute_control", "Execute control iteration"),
    ]

    async def set_bot_commands(self):
        self.logger.info("Setting bot commands...")
        commands = [BotCommand(cmd, desc) for cmd, desc in self.COMMANDS]
        await self.app.bot.set_my_commands(commands)
        self.logger.info("Bot commands set!")

    def _setup_handlers(self):
        command_handlers = [
            CommandHandler(cmd, getattr(self, f"handle_{cmd}")) for cmd, _ in self.COMMANDS
        ]

        for handler in command_handlers:
            self.app.add_handler(handler)

        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_invalid))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

        self.app.add_error_handler(self.error_handler)
        self.logger.info("Handlers have been set up.")

    async def handle_help(self, update: Update, context: CallbackContext):
        help_text = "Available Commands:\n\n" + "\n".join(f"/{cmd} - {desc}" for cmd, desc in self.COMMANDS)
        await update.message.reply_text(help_text)

    @staticmethod
    async def handle_get_id(update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        await update.message.reply_text(f"{user_id}")
        context.bot.logger.info(f"User {user_id} requested their ID.")

    @authorized_only
    async def handle_set_controller_status(self, update: Update, context: CallbackContext):
        self.logger.info(f"User {update.message.from_user.id} is setting controller status.")
        keyboard = [[InlineKeyboardButton("Active", callback_data="set_controller_active"),
                     InlineKeyboardButton("Passive", callback_data="set_controller_passive")]]
        await update.message.reply_text("Choose the controller status:", reply_markup=InlineKeyboardMarkup(keyboard))

    @authorized_only
    async def handle_set_battery_mode_direct(self, update: Update, context: CallbackContext):
        self.logger.info(f"User {update.message.from_user.id} is setting controller status.")
        keyboard = [[InlineKeyboardButton("FULLY_FEED_TO_GRID", callback_data="set_fftg"),
                     InlineKeyboardButton("MAXIMUM_SELF_CONSUMPTION", callback_data="set_msf")]]
        await update.message.reply_text("Choose the battery mode:", reply_markup=InlineKeyboardMarkup(keyboard))

    @authorized_only
    async def handle_get_controller_status(self, update: Update, context: CallbackContext):
        status = "Active" if self.system.get_active() else "Passive"
        self.logger.info(f"User {update.message.from_user.id} requested controller status: {status}.")
        await update.message.reply_text(f"Controller Status: {status}")

    @authorized_only
    async def handle_get_battery_mode(self, update: Update, context: CallbackContext):
        mode = self.system.get_last_battery_mode()
        self.logger.info(f"User {update.message.from_user.id} requested battery mode: {mode}.")
        await update.message.reply_text(f"Current Battery Mode: {mode}")

    @authorized_only
    async def handle_get_battery_mode_direct(self, update: Update, context: CallbackContext):
        mode = self.system.get_real_battery_mode().name
        self.logger.info(f"User {update.message.from_user.id} requested real battery mode: {mode}.")
        await update.message.reply_text(f"Real Current Battery Mode: {mode}")

    @authorized_only
    async def handle_execute_control(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        self.logger.info(f"User {user_id} initiated control execution.")
        await update.message.reply_text("Execution started...")
        try:
            self.system.execute_control()
            result = self.system.get_last_battery_mode()
            self.logger.info(f"Execution control completed successfully. Result: {result}.")
            await update.message.reply_text(f"Execution control result: {result}")
        except FusionSolarExceptionExtended as e:
            self.logger.error(f"Execution control failed for user {user_id}: {e.code}")
            await update.message.reply_text(f"Execution control failed: {e.code}")


    @authorized_only
    async def handle_invalid(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        self.logger.info(f"User {user_id} entered an invalid command.")
        await update.message.reply_text("Invalid command! Use '/help' for a list of commands.")

    async def button_callback(self, update: Update, context: CallbackContext):
        query = update.callback_query
        user_id = query.from_user.id
        action = query.data

        self.logger.info(f"User {user_id} triggered button callback: {action}.")

        if not await self._handle_unauthorized_access(query, user_id):
            return

        actions = {
            "set_controller_active": lambda: self._set_controller_state(query, user_id, True),
            "set_controller_passive": lambda: self._set_controller_state(query, user_id, False),
            "set_fftg": lambda: self._set_battery_mode(query, user_id, BatteryWorkingMode.FULLY_FEED_TO_GRID),
            "set_msf": lambda: self._set_battery_mode(query, user_id, BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION),
        }

        if action in actions:
            await actions[action]()
        else:
            self.logger.warning(f"User {user_id} sent an unknown action: {action}.")
            await query.answer("Unknown command!", show_alert=True)

    # Helper method for unauthorized access handling
    async def _handle_unauthorized_access(self, query: CallbackQuery, user_id: int):
        if not self._is_authorized(user_id):
            self.logger.info(f"Unauthorized user {user_id} attempted to use a button callback.")
            await query.answer("Unauthorized access!", show_alert=True)
            return False
        await query.answer()
        return True

    # Helper method for setting controller state
    async def _set_controller_state(self, query: CallbackQuery, user_id: int, state: bool):
        self.system.set_active(state)
        state_text = "Active" if state else "Passive"
        self.logger.info(f"User {user_id} set controller status to {state_text}.")
        await query.edit_message_text(f"Controller status set to {state_text}!")

    # Helper method for setting battery mode
    async def _set_battery_mode(self, query: CallbackQuery, user_id: int, battery_mode: BatteryWorkingMode):
        system_client = self.system.client
        self.logger.info(f"User {user_id} is changing battery mode to {battery_mode.name}.")

        try:
            await query.edit_message_text(f"Changing battery mode to {battery_mode.name}...")
            system_client.set_battery_working_mode(self.system.battery_id, battery_mode)
            self.logger.info(f"User {user_id} set battery working mode to {battery_mode.name}.")
            await query.edit_message_text(f"Battery working mode set to {battery_mode.name}.")
        except FusionSolarExceptionExtended as e:
            self.logger.error(f"Failed to set battery mode for user {user_id}: {e.code}")
            await query.edit_message_text(f"Failed to set battery mode: {e.code}")

    def run(self):
        self.logger.info("Bot is starting...")
        self.app.run_polling()
        self.logger.info("Bot has stopped.")
