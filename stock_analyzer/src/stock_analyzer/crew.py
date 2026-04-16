import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from stock_analyzer.tools.data_fetcher import StockDataFetcher
from stock_analyzer.tools.indicators import TechnicalIndicatorsTool
from stock_analyzer.tools.patterns import ChartPatternsTool

load_dotenv()

llm = LLM(
    model="openrouter/auto",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# initialize tools
stock_data_tool = StockDataFetcher()
indicators_tool = TechnicalIndicatorsTool()
patterns_tool = ChartPatternsTool()

@CrewBase
class StockAnalyzer():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def data_fetcher(self) -> Agent:
        return Agent(
            config=self.agents_config['data_fetcher'],
            llm=llm,
            tools=[
                stock_data_tool,
                indicators_tool,
                patterns_tool
            ],
            verbose=True
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'],
            llm=llm,
            verbose=True
        )

    @agent
    def signal_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['signal_generator'],
            llm=llm,
            verbose=True
        )

    @task
    def fetch_data_task(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_data_task'],
            agent=self.data_fetcher()
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            agent=self.technical_analyst(),
            context=[self.fetch_data_task()]
        )

    @task
    def signal_task(self) -> Task:
        return Task(
            config=self.tasks_config['signal_task'],
            agent=self.signal_generator(),
            context=[self.fetch_data_task(),
                     self.analysis_task()],
            output_file="output/signal_report.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )