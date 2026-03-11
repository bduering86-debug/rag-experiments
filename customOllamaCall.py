# Einfacher Test eines Ollama Calls
from ollama import Client
from rag_csv.config.settings import OllamaConfig
import sys

class CustomOllamaCall:

    def __init__(self, question, runs=1, profile='low', ollama_url=None):
        """
        Konstruktor: Wird beim Erstellen eines Objekts automatisch aufgerufen.
        :param question: Eingabestring für den Test (str)
        :param runs: Anzahl der Durchläufe (int)
        :param profile: Profil für den Ollama Call (str)
        :param ollama_url: Optionale benutzerdefinierte URL für den Ollama Call (str)
        """
        if not isinstance(question, str):
            raise TypeError("question muss ein String sein.")
        if not isinstance(runs, int):
            raise TypeError("runs muss eine Zahl sein.")
        if not isinstance(profile, str):
            raise TypeError("profile muss ein String sein.")
        if ollama_url is not None and not isinstance(ollama_url, str):
            raise TypeError("ollama_url muss ein String sein, wenn angegeben.")
        
        self.profile = (profile or "low").strip().lower()
        self.question = question
        self.runs = runs
        self.ollama_url = ollama_url.strip() if isinstance(ollama_url, str) else None

    def get_question(self):
        """Gibt die Frage zurück."""
        print(f"Question: {self.question}, Runs: {self.runs}")

    def get_runs(self):
        """Gibt die Anzahl der Durchläufe zurück."""
        return self.runs
    
    def get_profile(self):
        """Gibt das Profil zurück."""
        return self.profile

    def get_ollama_url(self):
        """Gibt die URL zurück, entweder benutzerdefiniert oder basierend auf dem Profil."""
        if self.ollama_url:
            url = self.ollama_url
            if "://" not in url:
                url = f"http://{url}"
            if ":" not in url.rsplit(":", 1)[-1]:
                url = f"{url}:11434"
            return url

        profile_urls = {
            "low": OllamaConfig.url_low_profile,
            "mid": OllamaConfig.url_mid_profile,
            "high": OllamaConfig.url_high_profile,
            "ultra": OllamaConfig.url_ultra_profile,
        }

        if self.profile not in profile_urls:
            raise ValueError(
                f"Ungültiges Profil '{self.profile}'. Profil muss 'low', 'mid', 'high' oder 'ultra' sein, "
                "oder eine benutzerdefinierte URL muss angegeben werden."
            )

        host = profile_urls[self.profile]
        if not host:
            raise ValueError(
                f"Keine URL für Profil '{self.profile}' gesetzt (siehe .env OLLAMA_URL_*)."
            )
        return host

    def __str__(self):
        """String-Darstellung des Objekts."""
        return f"Question: {self.question}, Runs: {self.runs}, Profile: {self.profile}"
    
    def call_ollama(self):
        profile = self.profile
        threads = 4

        if profile == 'low':
            threads = OllamaConfig.threads_low
        elif profile == 'mid':
            threads = OllamaConfig.threads_mid
        elif profile == 'high':
            threads = OllamaConfig.threads_high

        host = self.get_ollama_url()
        client = Client(host=host)
        
        response = client.chat(
            model="llama3.1:8b-instruct-q4_K_M",
            #model="llama3.1:8b-instruct-q6_K",
            messages=[
                {
                    "role": "user",
                    "content": self.prompt_builder()
                }
            ],
            options={
                "num_threads": threads,
                "num_ctx": OllamaConfig.num_ctx
            }
        )

        return response

    def run(self):
        """Führt den Test durch und gibt die Ergebnisse aus."""
        responses = []

        print(f"Starte Ollama Calls mit Profil '{self.profile}' und URL '{self.get_ollama_url()}'...")
        print(f"Kontextlänge: {OllamaConfig.num_ctx}")

        while self.runs > 0:
            print(f"Teste ollama-Call - Durchlauf {self.runs}...")
            response = self.call_ollama()
            print(f"Ollama Response: {response}")
            responses.append(response)
            self.runs -= 1

        return responses

    def prompt_builder(self):
        """Erstellt den Prompt für den Ollama Call."""
        prompt = f"""Du bis ein universeller Assistent und beantwortest mir nach bestem gewissen meine Fragen
            Verzichte dabei auf Floskeln und Einleitungen, sondern beantworte direkt die Frage.
             Bitte beantworte die Frage so ausführlich wie möglich und liefere relevante Informationen, Beispiele. Beschränke deine Antwort auf Maximal 200 Wörter. Wenn du die Frage nicht beantworten kannst, gib das bitte kurz und direkt an. """
        prompt += f"\nFrage: {self.question}"
        prompt += f"\nBitte beantworte die Frage so ausführlich wie möglich und liefere relevante Informationen, Beispiele"
        return prompt

    def get_token_metrics(self, response):
        """Extrahiert Token-Metriken aus der Ollama-Response."""
        
        if hasattr(response, "model_dump"):
            response = response.model_dump()
    
        if not isinstance(response, dict):
            raise TypeError(f"Unerwarteter response-Typ: {type(response)}")
        
        prompt_tokens = response.get("prompt_eval_count")
        prompt_eval_duration_ns = response.get("prompt_eval_duration")
        prompt_tokens_per_second = None
        generated_tokens = response.get("eval_count")
        generated_tokens_duration_ns = response.get("eval_duration")
        generated_tokens_per_second = None
        total_tokens = None

        if prompt_tokens is not None and generated_tokens is not None:
            total_tokens = prompt_tokens + generated_tokens
        
        if generated_tokens is not None and generated_tokens_duration_ns is not None:
            generated_tokens_per_second = generated_tokens / (generated_tokens_duration_ns / 1e9)

        if prompt_tokens is not None and prompt_eval_duration_ns is not None:
            prompt_tokens_per_second = prompt_tokens / (prompt_eval_duration_ns / 1e9)

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_per_second": prompt_tokens_per_second,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": generated_tokens_per_second,
            "total_tokens": total_tokens,
        }

# Beispielverwendung

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Was ist RAG?"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    profile = (sys.argv[3].strip() if len(sys.argv) > 3 else "low") or "low"
    ollama_url = (sys.argv[4].strip() if len(sys.argv) > 4 else None) or None



    tester = CustomOllamaCall(question, runs, profile, ollama_url)
    responses = tester.run()
    
    for idx, resp in enumerate(responses, 1):
        metrics = tester.get_token_metrics(resp)
        resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
        message = resp_dict.get("message", {}) if isinstance(resp_dict, dict) else {}
        answer_text = message.get("content", "Keine Antwort")
        print(f"Run: {idx}")
        print(f"Prompt Tokens per second: {metrics['prompt_tokens_per_second']}")
        print(f"Generated Tokens: {metrics['generated_tokens']}")
        print(f"Token per second: {metrics['generated_tokens_per_second']}")
        print(f"Total Tokens: {metrics['total_tokens']}")
        print(f"Gesamtdauer (s): {resp_dict.get('total_duration', 0) / 1e9}")
        print(f"Answer: {answer_text}")
