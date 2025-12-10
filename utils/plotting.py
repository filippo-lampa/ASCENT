import matplotlib.pyplot as plt

def plot_mutant_prioritization_results(rewards, moving_average, v_losses, p_losses, networks_update_freq, sut_name,
                                       should_save=False, save_path=None, execution_id=None, parameters_set_id=None):

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', label='Individual Rewards', alpha=0.7)
    plt.plot(moving_average, color='orange', label='Average Reward', linewidth=2)

    # Labeling
    if parameters_set_id is not None:
        plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    if should_save and save_path:
        plt.savefig(save_path + '/rewards_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(v_losses)
    if parameters_set_id is not None:
        plt.title('Value Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Value Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/value_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(p_losses)
    if parameters_set_id is not None:
        plt.title('Policy Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Policy Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/policy_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()
