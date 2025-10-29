import irsim

env = irsim.make('otter_dynamic_random.yaml')   

for i in range(1000):

    env.step()
    env.render(0.05)

    if env.done():
        break

env.end()