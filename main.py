from chatarena.chatarena.chameleon_arena import ChameleonArena

arena = ChameleonArena.from_config("chatarena/examples/chameleon_copy.json")
arena.launch_cli(interactive=False, max_steps=20)
