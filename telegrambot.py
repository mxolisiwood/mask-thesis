import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from fastai.vision import load_learner

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

def start(update, context):
    update.message.reply_text(

        "EN : Please send me a photo of you and I will tell you if you're wearing a mask 😏 \n"

    )


def help_command(update, context):
    update.message.reply_text('My only purpose is to tell you if you are wearing a mask. Send a photo')




def load_model():
    global model
    model = load_learner('C:/Users/silab/Desktop/mask-thesis/2attempt.pkl')
    print('Model loaded')


def detect_mask(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')

    label = model.predict('user_photo.jpg')[0]
    if label == "with_mask":
        update.message.reply_text(
            "EN: Looks like you are wearing a mask 😷. I hope you don't forget it when going out!😉 \n\n"

        )
    else:
        update.message.reply_text(
            "EN: Looks like you are not wearing a mask 😷. Please wear one and stay safe 🙄\n\n"

        )

def main():
    load_model()
    updater = Updater(token="1499112744:AAHgq07V8Z8meug39XacJlPryLZbgOaLEoc", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    dp.add_handler(MessageHandler(Filters.photo, detect_mask))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()