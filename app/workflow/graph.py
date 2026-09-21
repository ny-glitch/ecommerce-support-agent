"""The fixed outer LangGraph workflow."""
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.workflow.agent import build_agent_graph
from app.workflow.nodes import WorkflowDependencies, build_nodes
from app.workflow.state import TurnRuntime, WorkflowState


def build_workflow(deps: WorkflowDependencies, checkpointer) -> CompiledStateGraph:
    graph = StateGraph(WorkflowState, context_schema=TurnRuntime)
    for name, node in build_nodes(deps).items():
        graph.add_node(name, node)
    graph.add_node('agent', build_agent_graph(deps.agent_dependencies))
    graph.add_edge(START, 'resolve')
    graph.add_edge('resolve', 'classify')
    graph.add_conditional_edges(
        'classify',
        lambda state: 'budget_reply' if state['budget_exhausted'] else state['route'],
        {'knowledge': 'retrieve', 'business': 'agent', 'complaint': 'complaint',
         'chitchat': 'chitchat', 'budget_reply': 'budget_reply'},
    )
    graph.add_conditional_edges(
        'retrieve',
        lambda state: 'budget_reply' if state['budget_exhausted'] else 'evidence_gate',
        {'budget_reply': 'budget_reply', 'evidence_gate': 'evidence_gate'},
    )
    graph.add_conditional_edges(
        'evidence_gate',
        lambda state: 'budget_reply' if state['budget_exhausted'] else state['knowledge_target'],
        {'workflow_answer': 'workflow_answer', 'agent_tools': 'agent',
         'agent_generate': 'agent', 'fallback': 'fallback',
         'budget_reply': 'budget_reply'},
    )
    for name in ('agent', 'workflow_answer', 'fallback', 'complaint',
                 'chitchat', 'budget_reply'):
        graph.add_edge(name, 'persist')
    graph.add_edge('persist', END)
    return graph.compile(checkpointer=checkpointer)


__all__ = ['WorkflowDependencies', 'build_nodes', 'build_workflow']
