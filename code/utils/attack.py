from attack.pgd import PGDL2, PGDLinf

def get_attack(attack_name, net, attack_criterion, mean, std, 
                pgd_step_size, pgd_epsilon, pgd_iterations, pgd_random_start):
    if attack_name == "pgd_l2":
        attack = PGDL2(model=net,
                        attack_criterion=attack_criterion,
                        mean=mean,
                        std=std,
                        step_size=pgd_step_size, 
                        epsilon=pgd_epsilon, 
                        iterations=pgd_iterations,
                        random_start=pgd_random_start)
    elif attack_name == "pgd_linf":
        attack = PGDLinf(model=net,
                        attack_criterion=attack_criterion,
                        mean=mean,
                        std=std,
                        step_size=pgd_step_size, 
                        epsilon=pgd_epsilon, 
                        iterations=pgd_iterations, 
                        random_start=pgd_random_start)
    else:
        raise ValueError('Unknown attack: {}'.format(attack_name))
    return attack