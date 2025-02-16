import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, MessageHandler, Application, filters, CallbackQueryHandler

from energymanagementrl.rl import EnergyManagementSystem

logger = logging.getLogger()


def start_bot(system: EnergyManagementSystem, token: str):
    async def handle_help(update: Update, context: CallbackContext):
        help_text = (
            "Available Commands:\n"
            "/help - Show this message\n"
            "/get_controller_status - Get the current controller status\n"
            "/set_controller_status - Set the controller's active status (choose Active or Inactive)\n"
            "/get_battery_mode - Get the current battery mode result\n"
            "/execute_control - Execute control iteration\n"
        )
        await update.message.reply_text(help_text)

    async def handle_set_controller_status(update: Update, context: CallbackContext):
        keyboard = [[InlineKeyboardButton("Active", callback_data="set_controller_active"),
                     InlineKeyboardButton("Inactive", callback_data="set_controller_inactive")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Choose the controller status:", reply_markup=reply_markup)

    async def handle_get_controller_status(update: Update, context: CallbackContext):
        await update.message.reply_text(f"Controller Status: {system.get_active()}")

    async def handle_get_battery_mode(update: Update, context: CallbackContext):
        await update.message.reply_text(f"Current Battery Mode: {system.get_last_battery_mode()}")

    async def handle_execute_control(update: Update, context: CallbackContext):
        await update.message.reply_text(f"Execution control: {system.execute_control()}")

    async def handle_invalid(update: Update, context: CallbackContext):
        await update.message.reply_text("Invalid command! Use '/help' for a list of commands.")

    # Dictionary to simulate the switch case
    command_map = {
        "/start": handle_help,
        "/help": handle_help,
        "/set_controller_status": handle_set_controller_status,
        "/get_battery_mode": handle_get_battery_mode,
        "/get_controller_status": handle_get_controller_status,
        "/execute_control": handle_execute_control,
    }

    async def handle_message(update: Update, context: CallbackContext):
        message = update.message.text.strip().lower()
        logger.warning('latest_message:' + message)

        # Get the handler from the command map or default to invalid command handler
        command_handler = command_map.get(message, handle_invalid)

        # Call the appropriate handler
        await command_handler(update, context)

    async def button_callback(update: Update, context: CallbackContext):
        query = update.callback_query
        await query.answer()

        if query.data == "set_controller_active":
            system.set_active(True)
            await query.edit_message_text("Controller status set to Active!")
        elif query.data == "set_controller_inactive":
            system.set_active(False)
            await query.edit_message_text("Controller status set to Inactive!")

    # Initialize the Telegram bot
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.COMMAND, handle_message))  # Use filters.COMMAND to capture commands
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_invalid)  # Filter out commands, accept general text
    )
    app.add_handler(CallbackQueryHandler(button_callback))
    app.run_polling()
