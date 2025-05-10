from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        from .vector_store import populate_vector_store_if_empty
        populate_vector_store_if_empty()
