# Typed Decisions

This service evaluates named questions against shared evidence and returns typed decisions.

## Language

**State**:
The evidence shared by every question in a decision request, expressed as text or structured data.
_Avoid_: Chat prompt, conversation

**Choice**:
A decision among named alternatives, accompanied by the selected alternative, its confidence, and a probability for every alternative.
_Avoid_: Generated answer

**Noul**:
The probability that a yes/no question is true, expressed as a number from zero to one.
_Avoid_: Boolean answer

**Score**:
The probability-weighted average of ordered rubric levels whose positions start at zero, accompanied by confidence, the rubric legend, and level probabilities.
_Avoid_: Winning level, integer rating

**Decider**:
The model family that evaluates typed questions through probabilities over explicitly supplied alternatives.
_Avoid_: Chat model, Jev

**Jev**:
TypeSafe's decision model family. Decider is a separate model family that can expose compatible typed decisions.
_Avoid_: Decider alias
