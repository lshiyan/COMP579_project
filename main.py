from chatarena.chatarena.arena import Arena

arena = Arena.from_config("chatarena/examples/chameleon.json")
arena.run(num_steps=10)

arena.launch_cli()
