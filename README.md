# Decomp_MPI
Code for decomposition driven communications built on MPI

This code implements decomposition driven communications on top of MPI.
Instead of coding communications directly in terms of messages you specify the data decompositions at the start and end of the communication phase.

For problems that are naturally expressed as changes in decomposition this is a significant saving. The same descriptors can bu used to drive MPI-IO or PGAS based communication.
