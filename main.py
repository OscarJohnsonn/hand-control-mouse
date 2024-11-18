
        # Limit the frame rate
        elapsed_time = time.time() - prev_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)
        prev_time = time.time()

        # Increment frame count
        frame_count += 1

        # Calculate and print FPS every second
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Show the frame
        # Uncomment the following line to display the video
        # cv2.imshow('Hand Tracking', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()