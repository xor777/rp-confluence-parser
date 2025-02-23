import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from knowledge_base import KnowledgeBase, Constants
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slack_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("slack_bolt").setLevel(logging.WARNING)

load_dotenv()

class SlackBot:
    def __init__(self):
        logger.info("=== Инициализация Slack бота ===")
        logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.bot_token = os.getenv('SLACK_BOT_TOKEN')
        if not self.bot_token:
            logger.error("SLACK_BOT_TOKEN не найден в переменных окружения")
            raise ValueError("SLACK_BOT_TOKEN не найден в переменных окружения")
            
        self.app_token = os.getenv('SLACK_APP_TOKEN')
        if not self.app_token:
            logger.error("SLACK_APP_TOKEN не найден в переменных окружения")
            raise ValueError("SLACK_APP_TOKEN не найден в переменных окружения")
            
        self.app = App(token=self.bot_token)
        self.kb = KnowledgeBase()
        
        logger.info("Инициализация базы знаний завершена")
        self.setup_handlers()
        logger.info("Обработчики событий настроены")
        logger.info("=== Инициализация завершена ===\n")
    
    def setup_handlers(self):
        @self.app.event("app_mention")
        def handle_mention(event, say):
            logger.info(f"упоминание: {event}")
            text = event['text']
            bot_user_id = self.app.client.auth_test()["user_id"]
            mention_pattern = f"<@{bot_user_id}>"
            question = text.replace(mention_pattern, '').strip()
            
            message_ts = event['ts']
            thread_ts = event.get('thread_ts', message_ts)
            
            logger.info(f"вопрос: '{question}' в треде: {thread_ts}, message_ts: {message_ts}")
            
            answer = self.get_answer_for_question(
                question, 
                event['channel'], 
                message_ts,
                thread_ts
            )
            
            say(
                text=answer, 
                thread_ts=thread_ts,
                parse='mrkdwn'
            )
            logger.info("ответ успешно отправлен")
        
        @self.app.event("message")
        def handle_direct_message(event, say):
            logger.info(f"сообщение: {event}")
            
            if event.get('bot_id'):
                logger.info("пропуск сообщения от бота")
                return
                
            if event.get('channel_type') != 'im':
                logger.info("пропуск сообщения не из личного чата")
                return
            
            question = event['text'].strip()
            message_ts = event['ts']
            logger.info(f"вопрос в личном чате: '{question}' с message_ts: {message_ts}")
            
            answer = self.get_answer_for_question(
                question, 
                event['channel'],
                message_ts
            )
            
            say(
                text=answer,
                parse='mrkdwn'
            )
            logger.info("ответ отправлен")
        
        @self.app.command("/ask")
        def handle_slash_command(ack, body, say):
            logger.info(f"получена команда: {body}")
            ack()
            
            question = body['text'].strip()
            message_ts = body['command_ts']
            thread_ts = body.get('thread_ts', message_ts)
            
            logger.info(f"команда: '{question}' в треде: {thread_ts}, message_ts: {message_ts}")
            
            answer = self.get_answer_for_question(
                question, 
                body['channel_id'], 
                message_ts,
                thread_ts
            )
            
            say(
                text=answer,
                thread_ts=thread_ts,
                parse='mrkdwn'
            )
            logger.info("ответ отправлен")
    
    def send_typing_indicator(self, channel: str, message_ts: str = None):
        try:
            if message_ts:
                logger.info(f"👀 в канале {channel} к сообщению {message_ts}")
                self.app.client.reactions_add(
                    channel=channel,
                    timestamp=message_ts,
                    name="eyes"
                )
                logger.info("👀 добавлен")
            else:
                logger.info("message_ts не предоставлен, пропускаем реакцию")
        except Exception as e:
            logger.error(f"ошибка отправки реакции: {str(e)}", exc_info=True)
    
    def get_answer_for_question(self, question: str, channel: str, message_ts: str = None, thread_ts: str = None) -> str:
        if not question:
            logger.info("пустой вопрос, отправка приветственного сообщения")
            return Constants.WELCOME_MESSAGE
            
        logger.info(f"получаем ответ на вопрос: '{question}' в канале: {channel}, message_ts: {message_ts}, thread_ts: {thread_ts}")
        
        try:
            self.send_typing_indicator(channel, message_ts)
            answer = self.kb.get_answer(question)
            logger.info("ответ получен")
            return answer
        except Exception as e:
            logger.error(f"ошибка получения ответа: {str(e)}", exc_info=True)
            return Constants.TECHNICAL_ERROR_MESSAGE
    
    def run(self):
        logger.info("запуск slack бота...")
        try:
            handler = SocketModeHandler(self.app, self.app_token)
            handler.start()
            logger.info("бот запущен")
        except Exception as e:
            logger.error(f"не удалось запустить slack бот: {str(e)}")
            raise

def main():
    start_time = datetime.now()
    logger.info(f"Начало работы бота: {start_time}")
    
    try:
        bot = SlackBot()
        bot.run()
    except Exception as e:
        logger.error("Критическая ошибка при работе бота")
        logger.exception(str(e))
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Завершение работы бота: {end_time}")
        logger.info(f"Общее время работы: {duration}")

if __name__ == "__main__":
    main() 