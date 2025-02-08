import os
from dotenv import load_dotenv
from openai import OpenAI

from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Load environment variables
load_dotenv()

# Validate API Key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define the evaluation metric
metric = TaskCompletionMetric(
    threshold=0.8,
    model="gpt-4o",
    include_reason=True
)

# ✅ New Advanced Test Case
test_case = LLMTestCase(
    input="Plan a 7-day luxury trip to Italy, covering history, art, food, and local experiences.",
    actual_output=(
        "Day 1: Arrive in Rome, private tour of the Colosseum, dinner at La Pergola. "
        "Day 2: Vatican City tour, Sistine Chapel, traditional Roman cuisine at Roscioli. "
        "Day 3: Travel to Florence, Uffizi Gallery, wine tasting in Tuscany. "
        "Day 4: Visit Pisa, Leaning Tower, local seafood experience. "
        "Day 5: Venice arrival, gondola ride, dinner overlooking the Grand Canal. "
        "Day 6: Explore Milan, fashion district, opera night at La Scala. "
        "Day 7: Relax in Lake Como, spa experience, return flight."
    ),
    tools_called=[
        ToolCall(
            name="Itinerary Generator",
            description="Creates personalized travel itineraries based on interests, duration, and travel preferences.",
            input_parameters={"destination": "Italy", "days": 7, "theme": "Luxury"},
            output=[
                "Day 1: Rome - Colosseum private tour, La Pergola dinner.",
                "Day 2: Vatican, Sistine Chapel, Roscioli dining.",
                "Day 3: Florence - Uffizi Gallery, Tuscany wine tour.",
                "Day 4: Pisa - Leaning Tower, seafood tasting.",
                "Day 5: Venice - Gondola ride, Grand Canal dinner.",
                "Day 6: Milan - Fashion district, La Scala opera.",
                "Day 7: Lake Como - Spa retreat, return flight.",
            ],
        ),
        ToolCall(
            name="Flight Scheduler",
            description="Finds the best flights based on budget, departure city, and preferences.",
            input_parameters={"from": "New York", "to": "Rome", "class": "Business", "budget": 3000},
            output=["Direct Flight: New York (JFK) → Rome (FCO), Business Class, $2800"],
        ),
        ToolCall(
            name="Hotel Recommender",
            description="Suggests top-rated hotels based on luxury, budget, or boutique preferences.",
            input_parameters={"city": "Rome", "preference": "Luxury"},
            output=["Hotel Hassler Roma - 5-star, near Spanish Steps, $750 per night"],
        ),
        ToolCall(
            name="Weather Forecaster",
            description="Provides weather forecasts for the given travel dates.",
            input_parameters={"city": "Rome", "dates": "March 10-17"},
            output=["Sunny, 18°C to 22°C throughout the week."],
        ),
        ToolCall(
            name="Local Event Finder",
            description="Finds concerts, art exhibitions, and cultural events happening during the stay.",
            input_parameters={"city": "Florence", "dates": "March 12-13"},
            output=["Michelangelo Exhibit at Uffizi, Wine & Food Festival at Piazza della Signoria"],
        ),
    ],
)

# Evaluate the test case
metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")

# Evaluate multiple test cases
evaluate([test_case], [metric])