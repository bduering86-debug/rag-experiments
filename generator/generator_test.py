# generator_test.py

from ticket_generator import TicketGenerator

# Liste der Modelle die getestet werden sollen
MODELS = [
    "gpt-4o-mini",
    "llama3",
    "mistral-nemo",
]

# Todo: einbinden und testen von weiteren Modellen
def run_tests(num_tickets: int):
    for model in MODELS:
        print("=" * 80)
        print(f"Starte Testreihe für Modell: {model}")
        print("=" * 80)

        generator = TicketGenerator(
            model=model,
            num_tickets=num_tickets,
            output_prefix=model
        )

        results = generator.run()

        print(f"Modell {model} erzeugte {len(results)} Tickets.\n")

if __name__ == "__main__":
    # Beispielwerte
    run_tests(
        num_tickets=5
    )
