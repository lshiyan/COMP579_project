from chatarena.chatarena.chameleon_arena import ChameleonArena

arena = ChameleonArena.from_config("chatarena/examples/chameleon_copy.json")
arena.run(num_steps=10)

arena.launch_cli()
