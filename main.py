from chatarena.experiments.cs_experiment import ClosedSourceExperiment

cs_experiment = ClosedSourceExperiment("chatarena/examples/chameleon_closed_3p.json", "openai")
cs_experiment.run()
