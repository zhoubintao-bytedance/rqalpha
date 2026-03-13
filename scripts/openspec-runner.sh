#!/bin/bash

# OpenSpec Task Runner - Automated task execution for OpenSpec changes
# Inspired by continuous-claude, designed for OpenSpec workflow integration

VERSION="v0.1.0"

# ============================================================================
# Configuration
# ============================================================================

# Claude flags
ADDITIONAL_FLAGS="--dangerously-skip-permissions"

# Default settings
NOTES_FILE="OPENSPEC_NOTES.md"
MAX_RUNS=""
MAX_COST=""
MAX_DURATION=""
CHANGE_ID=""
TASKS_FILE=""
DRY_RUN=false
VERBOSE=false
COMPLETION_SIGNAL="OPENSPEC_CHANGE_COMPLETE"
COMPLETION_THRESHOLD=2

# State tracking
total_cost=0
successful_iterations=0
failed_iterations=0
total_iterations=0
consecutive_failures=0
MAX_CONSECUTIVE_FAILURES=3
completion_signal_count=0
start_time=""
current_task_index=0

# ============================================================================
# Prompt Templates
# ============================================================================

PROMPT_WORKFLOW_CONTEXT='## OpenSpec Task Runner Context

You are working on an OpenSpec change proposal. This is an automated loop that executes tasks one at a time from the tasks.md file.

**Key Points:**
1. Focus on completing ONE task at a time
2. After completing a task, make a git commit with a clear message describing the changes
3. The system will automatically update tasks.md and move to the next task
4. Follow the TDD workflow specified in the tasks.md if applicable
5. Reference the proposal.md, design.md, and spec.md files for context

**Completion Signal:** If the current task is fully complete, include "TASK_COMPLETE" in your response.
If the ENTIRE change (all tasks) is complete, include "COMPLETION_SIGNAL_PLACEHOLDER" in your response.

## Current OpenSpec Change'

PROMPT_TASK_EXECUTION='## Current Task

You are working on the following task from the change proposal:

**Task:** TASK_PLACEHOLDER

**Instructions:**
1. Read the relevant spec files in `openspec/changes/CHANGE_ID_PLACEHOLDER/specs/` for requirements
2. If this is a TDD task, follow Red → Green → Refactor cycle
3. Implement the task following existing patterns in the codebase
4. Run tests if applicable
5. **Make a git commit** with a clear, concise message describing the completed work
   - Use format: `git add . && git commit -m "feat: <description>"`
   - Commit message should explain what was done for this task
6. Once complete, output "TASK_COMPLETE" to signal completion

## Additional Context'

PROMPT_NOTES_UPDATE='## Session Notes

Update the notes file with:
- What was accomplished in this iteration
- Any blockers or issues encountered
- Context for the next iteration

Keep notes concise and actionable.'

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo "ℹ️  $1" >&2
}

log_success() {
    echo "✅ $1" >&2
}

log_warning() {
    echo "⚠️  $1" >&2
}

log_error() {
    echo "❌ $1" >&2
}

log_debug() {
    if [ "$VERBOSE" = "true" ]; then
        echo "🔍 $1" >&2
    fi
}

parse_duration() {
    local duration_str="$1"
    duration_str=$(echo "$duration_str" | tr -d '[:space:]')

    if [ -z "$duration_str" ]; then
        return 1
    fi

    local total_seconds=0
    local remaining="$duration_str"

    if [[ "$remaining" =~ ([0-9]+)[hH] ]]; then
        local hours="${BASH_REMATCH[1]}"
        total_seconds=$((total_seconds + hours * 3600))
        remaining="${remaining/${BASH_REMATCH[0]}/}"
    fi

    if [[ "$remaining" =~ ([0-9]+)[mM] ]]; then
        local minutes="${BASH_REMATCH[1]}"
        total_seconds=$((total_seconds + minutes * 60))
        remaining="${remaining/${BASH_REMATCH[0]}/}"
    fi

    if [[ "$remaining" =~ ([0-9]+)[sS] ]]; then
        local seconds="${BASH_REMATCH[1]}"
        total_seconds=$((total_seconds + seconds))
        remaining="${remaining/${BASH_REMATCH[0]}/}"
    fi

    if [ -n "$remaining" ]; then
        return 1
    fi

    if [ $total_seconds -eq 0 ]; then
        return 1
    fi

    echo "$total_seconds"
    return 0
}

format_duration() {
    local seconds="$1"

    if [ -z "$seconds" ] || [ "$seconds" -eq 0 ]; then
        echo "0s"
        return
    fi

    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    local result=""
    if [ $hours -gt 0 ]; then
        result="${hours}h"
    fi
    if [ $minutes -gt 0 ]; then
        result="${result}${minutes}m"
    fi
    if [ $secs -gt 0 ] || [ -z "$result" ]; then
        result="${result}${secs}s"
    fi

    echo "$result"
}

# ============================================================================
# Task Parsing Functions
# ============================================================================

# Parse tasks.md and extract all tasks with their status
parse_tasks_file() {
    local tasks_file="$1"

    if [ ! -f "$tasks_file" ]; then
        log_error "Tasks file not found: $tasks_file"
        return 1
    fi

    # Extract tasks with format: - [ ] or - [x]
    grep -n "^- \[" "$tasks_file" | while IFS= read -r line; do
        local line_num=$(echo "$line" | cut -d: -f1)
        local task_content=$(echo "$line" | cut -d: -f2-)
        local is_done="false"

        if echo "$task_content" | grep -q "^\- \[x\]"; then
            is_done="true"
        fi

        # Extract task description (remove checkbox)
        local task_desc=$(echo "$task_content" | sed 's/^- \[[x ]\] //')

        echo "${line_num}|${is_done}|${task_desc}"
    done
}

# Get the next uncompleted task
get_next_task() {
    local tasks_file="$1"

    parse_tasks_file "$tasks_file" | while IFS='|' read -r line_num is_done task_desc; do
        if [ "$is_done" = "false" ]; then
            echo "${line_num}|${task_desc}"
            return 0
        fi
    done
}

# Count total and completed tasks
count_tasks() {
    local tasks_file="$1"
    local total=0
    local completed=0

    while IFS='|' read -r line_num is_done task_desc; do
        total=$((total + 1))
        if [ "$is_done" = "true" ]; then
            completed=$((completed + 1))
        fi
    done < <(parse_tasks_file "$tasks_file")

    echo "${completed}/${total}"
}

# Mark a task as complete in tasks.md
mark_task_complete() {
    local tasks_file="$1"
    local line_num="$2"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "(DRY RUN) Would mark task at line $line_num as complete"
        return 0
    fi

    # Use sed to replace [ ] with [x] on the specific line
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "${line_num}s/\- \[ \]/- [x]/" "$tasks_file"
    else
        sed -i "${line_num}s/\- \[ \]/- [x]/" "$tasks_file"
    fi

    log_success "Task marked as complete in $tasks_file"
}

# ============================================================================
# OpenSpec Context Functions
# ============================================================================

# Read OpenSpec context files
get_openspec_context() {
    local change_id="$1"
    local base_path="openspec/changes/${change_id}"
    local context=""

    # Read proposal.md
    if [ -f "${base_path}/proposal.md" ]; then
        context+="### Proposal Summary\n"
        context+=$(head -50 "${base_path}/proposal.md")
        context+="\n\n"
    fi

    # Read design.md if exists
    if [ -f "${base_path}/design.md" ]; then
        context+="### Design Decisions\n"
        context+=$(head -50 "${base_path}/design.md")
        context+="\n\n"
    fi

    # List spec files
    if [ -d "${base_path}/specs" ]; then
        context+="### Spec Files\n"
        context+=$(find "${base_path}/specs" -name "*.md" -type f)
        context+="\n"
    fi

    echo -e "$context"
}

# ============================================================================
# Claude Execution
# ============================================================================

build_prompt() {
    local change_id="$1"
    local task_desc="$2"
    local task_progress="$3"

    local prompt="${PROMPT_WORKFLOW_CONTEXT//COMPLETION_SIGNAL_PLACEHOLDER/$COMPLETION_SIGNAL}"

    prompt+="\n\n**Change ID:** ${change_id}"
    prompt+="\n**Progress:** ${task_progress}"
    prompt+="\n\n"

    # Add task execution prompt
    local task_prompt="${PROMPT_TASK_EXECUTION//TASK_PLACEHOLDER/$task_desc}"
    task_prompt="${task_prompt//CHANGE_ID_PLACEHOLDER/$change_id}"
    prompt+="$task_prompt"

    # Add OpenSpec context
    prompt+="\n\n"
    prompt+=$(get_openspec_context "$change_id")

    # Add notes if exists
    if [ -f "$NOTES_FILE" ]; then
        prompt+="\n\n### Previous Session Notes\n"
        prompt+=$(cat "$NOTES_FILE")
    fi

    # Add notes update instruction
    prompt+="\n\n$PROMPT_NOTES_UPDATE"

    echo -e "$prompt"
}

run_claude_iteration() {
    local prompt="$1"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "(DRY RUN) Would run Claude Code with prompt"
        echo '{"result": "DRY RUN - Task completed successfully. TASK_COMPLETE", "total_cost_usd": 0.05}'
        return 0
    fi

    # Show prompt in verbose mode
    if [ "$VERBOSE" = "true" ]; then
        echo "" >&2
        log_debug "Prompt being sent to Claude:"
        echo "┌─────────────────────────────────────────────────────────────┐" >&2
        echo "$prompt" | head -50 >&2
        if [ $(echo "$prompt" | wc -l) -gt 50 ]; then
            echo "... (truncated, total $(echo "$prompt" | wc -l) lines)" >&2
        fi
        echo "└─────────────────────────────────────────────────────────────┘" >&2
    fi

    log_info "Running Claude Code..."
    echo "═══════════════════════════════════════════════════════════════" >&2

    # Temporary file to capture output
    local temp_file=$(mktemp)

    # Run claude: display output to stderr (for user to see), capture to temp file
    claude -p "$prompt" $ADDITIONAL_FLAGS --output-format json 2>&1 | tee "$temp_file" >&2
    local exit_code=${PIPESTATUS[0]}

    echo "═══════════════════════════════════════════════════════════════" >&2

    if [ $exit_code -ne 0 ]; then
        log_error "Claude Code failed with exit code: $exit_code"
        rm -f "$temp_file"
        return 1
    fi

    # Extract result from captured output (last valid JSON array or object)
    local result=$(cat "$temp_file" | grep -E '^\[?\{' | tail -1)
    rm -f "$temp_file"

    # Output result to stdout for caller to capture
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo '{"result": "Task executed", "total_cost_usd": 0}'
    fi

    return 0
}

# ============================================================================
# Main Loop
# ============================================================================

execute_single_task() {
    local change_id="$1"
    local tasks_file="$2"
    local iteration=$3

    # Get next task
    local next_task=$(get_next_task "$tasks_file")

    if [ -z "$next_task" ]; then
        log_success "All tasks completed!"
        return 2  # Signal all done
    fi

    local line_num=$(echo "$next_task" | cut -d'|' -f1)
    local task_desc=$(echo "$next_task" | cut -d'|' -f2-)
    local task_progress=$(count_tasks "$tasks_file")

    echo "" >&2
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Iteration $iteration | Progress: $task_progress"
    log_info "Current Task: $task_desc"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build prompt
    local prompt=$(build_prompt "$change_id" "$task_desc" "$task_progress")

    log_debug "Prompt built, executing Claude Code..."

    # Run Claude
    local result
    result=$(run_claude_iteration "$prompt")
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        failed_iterations=$((failed_iterations + 1))
        consecutive_failures=$((consecutive_failures + 1))
        log_error "Task execution failed (consecutive failures: $consecutive_failures)"
        return 1
    fi

    # Parse result
    local result_text=$(echo "$result" | jq -r 'if type == "array" then .[-1].result // empty else .result // empty end' 2>/dev/null)
    local cost=$(echo "$result" | jq -r 'if type == "array" then .[-1].total_cost_usd // 0 else .total_cost_usd // 0 end' 2>/dev/null)

    if [ -n "$cost" ] && [ "$cost" != "null" ]; then
        total_cost=$(awk "BEGIN {printf \"%.3f\", $total_cost + $cost}")
        printf "💰 Cost: \$%.3f (Total: \$%.3f)\n" "$cost" "$total_cost" >&2
    fi

    # Check for task completion signal
    if echo "$result_text" | grep -q "TASK_COMPLETE"; then
        log_success "Task completed!"
        mark_task_complete "$tasks_file" "$line_num"
        successful_iterations=$((successful_iterations + 1))
        consecutive_failures=0  # Reset on success

        # Check for overall completion
        if echo "$result_text" | grep -q "$COMPLETION_SIGNAL"; then
            completion_signal_count=$((completion_signal_count + 1))
            log_info "Completion signal detected ($completion_signal_count/$COMPLETION_THRESHOLD)"
        else
            completion_signal_count=0
        fi

        return 0
    else
        consecutive_failures=$((consecutive_failures + 1))
        log_warning "Task not marked as complete, will retry (consecutive failures: $consecutive_failures)"
        return 1
    fi
}

check_limits() {
    # Check max runs (based on successful iterations)
    if [ -n "$MAX_RUNS" ] && [ "$MAX_RUNS" -gt 0 ] && [ $successful_iterations -ge "$MAX_RUNS" ]; then
        log_info "Max successful runs reached ($MAX_RUNS)"
        return 1
    fi

    # Check max cost
    if [ -n "$MAX_COST" ]; then
        if [ "$(awk "BEGIN {print ($total_cost >= $MAX_COST)}")" = "1" ]; then
            log_info "Max cost reached (\$$MAX_COST)"
            return 1
        fi
    fi

    # Check max duration
    if [ -n "$MAX_DURATION" ] && [ -n "$start_time" ]; then
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        if [ $elapsed -ge "$MAX_DURATION" ]; then
            log_info "Max duration reached ($(format_duration $elapsed))"
            return 1
        fi
    fi

    # Check completion threshold
    if [ $completion_signal_count -ge $COMPLETION_THRESHOLD ]; then
        log_success "Completion threshold reached!"
        return 1
    fi

    # Check consecutive failures
    if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
        log_error "Too many consecutive failures ($MAX_CONSECUTIVE_FAILURES)"
        return 1
    fi

    return 0
}

main_loop() {
    local change_id="$1"
    local tasks_file="$2"
    local iteration=1

    start_time=$(date +%s)

    log_info "Starting OpenSpec Task Runner"
    log_info "Change ID: $change_id"
    log_info "Tasks file: $tasks_file"
    log_info "Initial progress: $(count_tasks "$tasks_file")"

    while true; do
        # Check limits before each iteration
        if ! check_limits; then
            break
        fi

        # Execute single task
        execute_single_task "$change_id" "$tasks_file" $iteration
        local result=$?

        if [ $result -eq 2 ]; then
            # All tasks completed
            break
        fi

        # Small delay between iterations
        sleep 2
        iteration=$((iteration + 1))
    done

    # Summary
    echo "" >&2
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Session Complete"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Tasks completed: $successful_iterations"
    log_info "Tasks failed: $failed_iterations"
    log_info "Final progress: $(count_tasks "$tasks_file")"

    if [ -n "$start_time" ]; then
        local elapsed=$(($(date +%s) - start_time))
        log_info "Duration: $(format_duration $elapsed)"
    fi

    if [ "$(awk "BEGIN {print ($total_cost > 0)}")" = "1" ]; then
        printf "💰 Total cost: \$%.3f\n" "$total_cost" >&2
    fi
}

# ============================================================================
# CLI Interface
# ============================================================================

show_help() {
    cat << 'EOF'
OpenSpec Task Runner - Automated task execution for OpenSpec changes

USAGE:
    openspec-runner.sh --change <change-id> [options]
    openspec-runner.sh --tasks <tasks-file> --change <change-id> [options]

REQUIRED:
    -c, --change <id>          OpenSpec change ID (e.g., add-student-management-system)

OPTIONS:
    -t, --tasks <file>         Path to tasks.md file (auto-detected from change ID if not specified)
    -m, --max-runs <n>         Maximum number of successful task completions
    --max-cost <dollars>       Maximum cost in USD
    --max-duration <duration>  Maximum duration (e.g., "2h", "30m", "1h30m")
    --notes-file <file>        Notes file for context persistence (default: OPENSPEC_NOTES.md)
    --dry-run                  Simulate execution without making changes
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message

EXAMPLES:
    # Run all tasks for a change
    openspec-runner.sh --change add-student-management-system

    # Run with cost limit
    openspec-runner.sh --change add-student-management-system --max-cost 5.00

    # Run for specific duration
    openspec-runner.sh --change add-student-management-system --max-duration 1h

    # Run maximum 5 tasks
    openspec-runner.sh --change add-student-management-system --max-runs 5

    # Dry run to test
    openspec-runner.sh --change add-student-management-system --dry-run

WORKFLOW:
    1. Parses tasks.md to find uncompleted tasks
    2. Executes each task using Claude Code with full OpenSpec context
    3. Marks tasks as complete when Claude signals "TASK_COMPLETE"
    4. Updates notes file for context persistence
    5. Repeats until all tasks done or limits reached

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--change)
                CHANGE_ID="$2"
                shift 2
                ;;
            -t|--tasks)
                TASKS_FILE="$2"
                shift 2
                ;;
            -m|--max-runs)
                MAX_RUNS="$2"
                shift 2
                ;;
            --max-cost)
                MAX_COST="$2"
                shift 2
                ;;
            --max-duration)
                MAX_DURATION="$2"
                shift 2
                ;;
            --notes-file)
                NOTES_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

validate_arguments() {
    if [ -z "$CHANGE_ID" ]; then
        log_error "Change ID is required. Use --change <id>"
        exit 1
    fi

    # Auto-detect tasks file if not specified
    if [ -z "$TASKS_FILE" ]; then
        TASKS_FILE="openspec/changes/${CHANGE_ID}/tasks.md"
    fi

    if [ ! -f "$TASKS_FILE" ]; then
        log_error "Tasks file not found: $TASKS_FILE"
        exit 1
    fi

    # Validate duration format if specified
    if [ -n "$MAX_DURATION" ]; then
        local duration_seconds
        if ! duration_seconds=$(parse_duration "$MAX_DURATION"); then
            log_error "Invalid duration format: $MAX_DURATION"
            log_error "Use format like: 2h, 30m, 1h30m"
            exit 1
        fi
        MAX_DURATION="$duration_seconds"
    fi

    # Check if change directory exists
    local change_dir="openspec/changes/${CHANGE_ID}"
    if [ ! -d "$change_dir" ]; then
        log_error "Change directory not found: $change_dir"
        exit 1
    fi
}

validate_requirements() {
    # Skip Claude check in dry run mode
    if [ "$DRY_RUN" = "false" ]; then
        if ! command -v claude &> /dev/null; then
            log_error "Claude Code CLI is not installed"
            log_error "Install from: https://claude.ai/code"
            exit 1
        fi
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq is required for JSON parsing"
        log_error "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
        exit 1
    fi
}

main() {
    parse_arguments "$@"
    validate_arguments
    validate_requirements

    main_loop "$CHANGE_ID" "$TASKS_FILE"
}

main "$@"
