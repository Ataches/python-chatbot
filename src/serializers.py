
class MessageSerializer:
    @staticmethod
    def serialise(d) -> dict:
        return {
            "message": d.content,
            "id": d.id
        }
