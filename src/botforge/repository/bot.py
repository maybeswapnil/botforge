from typing import List, Optional

from models.bot import Bot

class BotRepository:
    def __init__(self, db_session):
        self.db = db_session

    def get_bot_by_id(self, bot_id: int) -> Optional[Bot]:
        return self.db.query(Bot).filter(Bot.id == bot_id).first()

    def get_all_bots(self) -> List[Bot]:
        return self.db.query(Bot).all()

    def create_bot(self, bot_data: dict) -> Bot:
        bot = Bot(**bot_data)
        self.db.add(bot)
        self.db.commit()
        self.db.refresh(bot)
        return bot

    def update_bot(self, bot_id: int, update_data: dict) -> Optional[Bot]:
        bot = self.get_bot_by_id(bot_id)
        if not bot:
            return None
        for key, value in update_data.items():
            setattr(bot, key, value)
        self.db.commit()
        self.db.refresh(bot)
        return bot

    def delete_bot(self, bot_id: int) -> bool:
        bot = self.get_bot_by_id(bot_id)
        if not bot:
            return False
        self.db.delete(bot)
        self.db.commit()
        return True