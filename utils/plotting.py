import matplotlib.pyplot as plt

def plot_mutant_prioritization_results(rewards, moving_average, v_losses, p_losses, o_losses, networks_update_freq, moving_average_window, sut_name):

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', label='Individual Rewards', alpha=0.7)
    plt.plot(moving_average, color='orange', label='Average Reward', linewidth=2)

    # Labeling
    plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name)
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    plt.plot(o_losses)
    plt.title('Observation Losses')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(v_losses)
    plt.title('Value Losses')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(p_losses)
    plt.title('Policy Losses')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    plt.show()
