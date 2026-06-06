import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

def research_squad(topic: str):

    # ─────────────────────────────────────────────────────────────
    # Agent 1: Research Analyst
    # ─────────────────────────────────────────────────────────────
    analyst = Agent(
        role="Research Analyst",
        goal="Gather comprehensive, accurate information on the given topic from reliable sources.",
        backstory=(
            "Seasoned research analyst with expertise in sourcing, "
            "evaluating, and structuring information from diverse domains."
        ),
        verbose=True,
        llm=llm
    )

    # ─────────────────────────────────────────────────────────────
    # Agent 2: Content Synthesizer
    # ─────────────────────────────────────────────────────────────
    synthesizer = Agent(
        role="Content Synthesizer",
        goal="Transform raw research notes into a clear, well-structured report.",
        backstory=(
            "Experienced science writer with a talent for distilling "
            "complex findings into engaging, readable narratives."
        ),
        verbose=True,
        llm=llm
    )

    # ─────────────────────────────────────────────────────────────
    # Agent 3: Fact Checker
    # ─────────────────────────────────────────────────────────────
    fact_checker = Agent(
        role="Fact Checker",
        goal="Verify the accuracy of all claims and flag anything unsupported or ambiguous.",
        backstory=(
            "Meticulous editor with 10+ years of fact-checking experience "
            "across journalism and academic publishing."
        ),
        verbose=True,
        llm=llm
    )

    # ─────────────────────────────────────────────────────────────
    # Task 1: Research
    # ─────────────────────────────────────────────────────────────
    research_task = Task(
        description=(
            f"Research the following topic thoroughly: '{topic}'. "
            "Identify key facts, statistics, notable developments, "
            "and credible sources. Organise findings under clear headings."
        ),
        expected_output=(
            "Structured research notes with 5–8 key findings, "
            "bullet-pointed evidence, and source references for each point."
        ),
        agent=analyst
    )

    # ─────────────────────────────────────────────────────────────
    # Task 2: Synthesis
    # ─────────────────────────────────────────────────────────────
    synthesis_task = Task(
        description=(
            "Using the research notes provided, write a cohesive report "
            "on the topic. Include an introduction, main findings section, "
            "and a conclusion with key takeaways."
        ),
        expected_output=(
            "A polished 400–600 word report with clear sections: "
            "Introduction, Key Findings, and Conclusion."
        ),
        agent=synthesizer,
        context=[research_task]
    )

    # ─────────────────────────────────────────────────────────────
    # Task 3: Fact-checking
    # ─────────────────────────────────────────────────────────────
    fact_check_task = Task(
        description=(
            "Review the synthesized report for factual accuracy. "
            "Cross-reference claims against the original research notes. "
            "Flag any unsupported statements, suggest corrections, "
            "and assign an overall confidence rating (High / Medium / Low)."
        ),
        expected_output=(
            "The final report with inline fact-check annotations, "
            "a list of any corrections made, and an overall confidence rating."
        ),
        agent=fact_checker,
        context=[synthesis_task]
    )

    # ─────────────────────────────────────────────────────────────
    # Crew
    # ─────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[analyst, synthesizer, fact_checker],
        tasks=[research_task, synthesis_task, fact_check_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


if __name__ == "__main__":
    topic = "The impact of large language models on software development"
    result = research_squad(topic)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)