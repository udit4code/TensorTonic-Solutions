from collections import deque, defaultdict 
import heapq as hq 

def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    # Assume that task_names are unique, so we can index them in a map for faster retrieval
    task_map = { }
    for task in tasks:
        task_map[task["name"]] = task
    # Step 2 : Build DAG using adjacency List representation  
    # Task u depends on Task v => So, edge (v, u) from v to u
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    for task in tasks:
        name = task["name"]
        if name not in in_degree:
            in_degree[name] = 0
        for parent in task["depends_on"]:
            graph[parent].append(name)
            in_degree[name] += 1
    # Step 3: Push all tasks into ready_queue whose indegree is 0.
    # Also, we get them sorted via task_name (as asked in the question)
    ready_queue = []
    for task_name, degree in in_degree.items():
        if degree == 0:
            hq.heappush(ready_queue, task_name)
    # Step 4 : Run Kahn's Topological sorting algorithm
    running_queue = []
    current_time = 0
    current_resources = 0
    schedule = []
    completed = set()
    started = set()
    total_tasks = len(tasks)
    while len(completed) < total_tasks:
        # Step 4.1 : Schedule tasks from ready_queue within the resource budget.
        # We make the transition from ready_queue to running.
        skipped = []
        while ready_queue:
            task_name = hq.heappop(ready_queue)
            task = task_map[task_name]
            needed = task["resources"]
            if current_resources + needed > resource_budget:
                skipped.append(task_name)
                continue
            started.add(task_name)
            schedule.append((task_name, current_time))
            current_resources += needed
            end_time = (current_time + task["duration"])
            hq.heappush(running_queue, (end_time, task_name))

        # Step 4.2 : Put skipped tasks back into ready_queue
        for task_name in skipped:
            hq.heappush(ready_queue, task_name)
        # Step 4.3 : Advance to next completion via Topological sort on running
        # So, we apply Topological sort on running_queue, not on ready_queue
        if not running_queue:
            # If not a single running task, then we are as good as done
            break
        next_time = running_queue[0][0]
        current_time = next_time
        # Step 4.4 : Complete all tasks ending now
        while running_queue and running_queue[0][0] == current_time:
            _, finished_task = hq.heappop(running_queue)
            completed.add(finished_task)
            # Relinquish the resources held by the completed task
            current_resources -= (task_map[finished_task]["resources"])
            # Step 4.5 : Now, for every running_task that can be ended, we end it and reduce the indegree of its child in the DAG
            for child in graph[finished_task]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    hq.heappush(ready_queue,child)

    # task_info = (task_name, timestamp_at_which_it_was_scheduled)
    return sorted(schedule, key=lambda task_info: (task_info[1], task_info[0]))