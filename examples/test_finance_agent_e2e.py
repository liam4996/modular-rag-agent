"""金融智能体端到端测试 — 10 scenarios. No real LLM needed."""
import os, ast
from langchain_core.runnables import RunnableLambda

def _mock_llm(): return RunnableLambda(lambda x: type("R",(),{"content":"{}"})())

def t1_agentstate():
    from src.agent.multi_agent.state import AgentState
    s=AgentState();s.user_input="test"
    s.add_to_blackboard("intent","financial_analysis","t")
    assert s.is_financial_intent
    s.add_to_blackboard("market_data",{"X":{}},"t");assert s.market_data=={"X":{}}
    s.add_to_blackboard("computed_results",{"m":{}},"t");assert s.computed_results=={"m":{}}
    s.add_to_blackboard("chart_paths",["a.png"],"t");assert s.chart_paths==["a.png"]
    s.add_to_blackboard("generated_report","R","t");assert s.generated_report=="R"
    s2=AgentState();s2.add_to_blackboard("intent","document_qa","t");assert not s2.is_financial_intent
    print("OK T1")

def t2_calculator():
    from src.agent.multi_agent.tools.financial_calculator import calculate_roe,calculate_roa,calculate_debt_to_asset,calculate_pe,calculate_pb,calculate_yoy,compute_all_metrics,FinancialMetrics
    assert calculate_roe(120,600)==20.0;assert calculate_roa(120,1000)==12.0
    assert calculate_debt_to_asset(400,1000)==40.0;assert calculate_pe(50,2.5)==20.0;assert calculate_pb(50,10)==5.0
    assert calculate_yoy(115,100)==15.0;assert calculate_yoy(None,100) is None
    m=compute_all_metrics({"net_income":120,"revenue":500,"total_equity":600,"total_assets":1000},{"price":50,"eps":2.5})
    assert isinstance(m,FinancialMetrics)
    d=m.to_dict();assert d["profitability"]["ROE"]==20.0;assert d["valuation"]["PE"]==20.0
    print("OK T2")

def t3_processor():
    from src.agent.multi_agent.tools.data_processor import build_comparison_table,rank_companies,detect_outliers,to_markdown_table,clean_financial_data,aggregate_market_to_table
    data={"A":{"PE":23,"ROE":18.2},"B":{"PE":35,"ROE":12.5},"C":{"PE":18,"ROE":22.1}}
    df=build_comparison_table(data,["PE","ROE"]);assert len(df)==3
    assert df[df["symbol"]=="A"]["PE"].values[0]==23.0
    ranked=rank_companies(df,"ROE");assert ranked["symbol"].values[0]=="C"
    assert detect_outliers(df,"PE")==[]
    md=to_markdown_table(df,"T");assert "A" in md
    raw=[{"symbol":"X","pe":"23.4"}];assert clean_financial_data(raw)["pe"].iloc[0]==23.4
    assert "quote" in aggregate_market_to_table({"quote":raw})
    print("OK T3")

def t4_visualizer():
    import pandas as pd
    from src.agent.multi_agent.tools.data_visualizer import DataVisualizer
    v=DataVisualizer("data/charts/e2e")
    df=pd.DataFrame({"date":["2024-01-02","2024-01-03","2024-01-04"],"open":[180,183,185],"high":[185,186,189],"low":[179,181,184],"close":[183,185,187],"volume":[1e6,1.2e6,0.9e6]})
    p=v.kline_chart(df,title="T",session_id="e2e",name="k");assert p.endswith(".png") and os.path.exists(p)
    p2=v.comparison_chart({"A":pd.DataFrame({"PE":[23]}),"B":pd.DataFrame({"PE":[35]})},"PE",title="T",session_id="e2e",name="c");assert p2.endswith(".png") and os.path.exists(p2)
    p3=v.pie_chart(["a","b"],[60,30],title="T",session_id="e2e",name="p");assert p3.endswith(".png") and os.path.exists(p3)
    assert v.kline_chart(pd.DataFrame())=="";assert v.comparison_chart({},"PE")=="";assert v.pie_chart([],[])==""
    print("OK T4")

def t5_renderer():
    from src.agent.multi_agent.tools.report_renderer import ReportRenderer
    from src.agent.multi_agent.tools.financial_calculator import compute_all_metrics
    rr=ReportRenderer()
    m=compute_all_metrics({"net_income":120,"revenue":500,"total_equity":600},{"price":50,"eps":2.5})
    r=rr.render_earnings_review(symbol="300750.SZ",metrics=m.to_dict(),valuation={"PE":"20.00"},profitability={"ROE":"20.00%"},growth={"revenue_yoy":"+15.0%"},key_findings=["F1"],charts=[{"title":"K","path":"c.png"}])
    assert "300750.SZ" in r and "PE" in r and "ROE" in r and "F1" in r and "Business Compute Agent" in r
    r2=rr.render_industry_comparison(symbols=["A","B"]);assert "A, B" in r2
    assert "财报分析报告" in rr.render_earnings_review()
    print("OK T5")

def t6_supervisor():
    from src.agent.multi_agent.supervisor_agent import SupervisorAgent
    from src.agent.multi_agent.state import AgentState
    sup=SupervisorAgent(_mock_llm())
    s=AgentState();s.user_input="test";s.add_to_blackboard("intent","financial_analysis","t");s.add_to_blackboard("needs_local",True,"r")
    sup.classify_and_plan(s);plan=s.task_plan
    assert len(plan["subtasks"])==3;assert plan["subtasks"][0]["type"]=="document_search";assert plan["subtasks"][1]["type"]=="financial_market";assert plan["subtasks"][2]["depends_on"]==["task_1"]
    assert not sup.all_completed(s);assert len(sup.get_next_subtasks(s))==2
    sup.mark_completed(s,"task_1");sup.mark_completed(s,"task_2");assert len(sup.get_next_subtasks(s))==1
    sup.mark_completed(s,"task_3");assert sup.all_completed(s)
    s2=AgentState();s2.user_input="q";s2.add_to_blackboard("intent","document_qa","t");sup.classify_and_plan(s2)
    assert len(s2.task_plan["subtasks"])==1;assert s2.task_plan["subtasks"][0]["type"]=="document_search"
    s3=AgentState();s3.add_to_blackboard("intent","chat","t");sup.classify_and_plan(s3);assert s3.task_plan is None
    sym=sup._extract_symbols("300750.SZ AAPL");assert len(sym)>0
    s4=AgentState();s4.add_to_blackboard("local_results",[{"t":1}],"t");s4.add_to_blackboard("market_data",{"q":[1]},"t")
    sup.aggregate(s4);agg=s4.blackboard["aggregated_context"];assert agg["local_docs"]==[{"t":1}];assert agg["market"]=={"q":[1]}
    print("OK T6")

def t7_business_compute():
    from src.agent.multi_agent import BusinessComputeAgent,AgentState
    s=AgentState();s.user_input="test";s.add_to_blackboard("intent","financial_analysis","t")
    s.add_to_blackboard("market_data",{"quote":[{"symbol":"300750.SZ","price":187.5}],"fundamentals":[{"symbol":"300750.SZ","pe":23.4,"pb":4.2,"roe":18.2,"gross_margin":22.4,"net_margin":11.5,"eps":3.8,"bvps":44.6},{"symbol":"300014.SZ","pe":35,"pb":5.1,"roe":12.5,"gross_margin":30,"net_margin":15,"eps":2.1,"bvps":25}]},"t")
    b=BusinessComputeAgent()
    b.compute(s,"calculate_metrics");cr=s.computed_results;assert"metrics"in cr and len(cr["metrics"])==2;assert cr["metrics"]["300750.SZ"]["valuation"]["PE"]==23.4
    b.compute(s,"compare_companies",{"metrics":["PE","ROE"]});assert"comparison_table"in s.computed_results;assert len(s.chart_paths)>=1
    b.compute(s,"generate_report");assert s.generated_report and len(s.generated_report)>100 and"Business Compute Agent"in s.generated_report
    print("OK T7")

def t8_router():
    from src.agent.multi_agent.router_agent import AgentType
    assert AgentType.FINANCE_DATA.value=="finance_data";assert AgentType.FINANCE_COMPUTE.value=="finance_compute"
    c=ast.dump(ast.parse(open("src/agent/multi_agent/router_agent.py",encoding="utf-8").read()))
    assert"financial_analysis"in c and"financial_market"in c and"financial_report"in c and"FinanceDataAgent"in c
    print("OK T8")

def t9_backward():
    from src.agent.multi_agent.supervisor_agent import SupervisorAgent
    from src.agent.multi_agent.state import AgentState
    sup=SupervisorAgent(_mock_llm())
    for intent,q in[("document_qa","q"),("summarization","s"),("comparison","c"),("analysis","a"),("chat","h")]:
        s=AgentState();s.user_input=q;s.add_to_blackboard("intent",intent,"t");sup.classify_and_plan(s)
        if intent=="chat":assert s.task_plan is None
        else:assert len(s.task_plan["subtasks"])==1 and s.task_plan["subtasks"][0]["type"]=="document_search"
    print("OK T9")

def t10_errors():
    from src.agent.multi_agent import BusinessComputeAgent,AgentState
    from src.agent.multi_agent.supervisor_agent import SupervisorAgent
    b=BusinessComputeAgent();sup=SupervisorAgent(_mock_llm())
    s=AgentState();s.user_input="t"
    try:b.compute(s,"calculate_metrics");b.compute(s,"compare_companies");b.compute(s,"generate_report");b.compute(s,"visualize")
    except Exception as e:assert False,f"crash: {e}"
    try:b.compute(AgentState(),"nonexistent")
    except:pass
    s3=AgentState();s3.add_to_blackboard("intent","unknown","t");sup.classify_and_plan(s3);assert s3.task_plan is None
    from src.agent.multi_agent.tools.financial_calculator import calculate_roe
    assert calculate_roe(float("nan"),600) is None;assert calculate_roe(100,float("nan")) is None
    from src.agent.multi_agent.tools.data_visualizer import DataVisualizer
    import pandas as pd
    v=DataVisualizer();assert v.kline_chart(pd.DataFrame())=="" and v.comparison_chart({},"PE")=="" and v.pie_chart([],[])==""
    print("OK T10")

if __name__=="__main__":
    print("="*50);print("Phase 4 E2E Tests");print("="*50)
    for t in[t1_agentstate,t2_calculator,t3_processor,t4_visualizer,t5_renderer,t6_supervisor,t7_business_compute,t8_router,t9_backward,t10_errors]:t()
    print("\n"+"="*50);print("All 10 tests passed!");print("="*50)
